#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ONNX to Neuropilot DLA Conversion Tool

This tool converts ONNX models to MediaTek Neuropilot DLA format.
Pipeline:
  1. Rewrite ONNX model from NCHW to NHWC format
  2. Output check to verify conversion correctness
  3. Convert ONNX to TFLite using mtk_onnx_converter
  4. Compile TFLite to DLA using ncc-tflite
  5. Generate EPContext wrapper ONNX model

NOTE: This tool only works on Linux, as the Neuropilot SDK tools
      (mtk_onnx_converter, ncc-tflite) are only available for Linux platforms.
"""

import os
import shutil
import subprocess
import argparse
import logging
import sys
from copy import deepcopy
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, shape_inference

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

try:
    import mtk_converter  # just for checking environments
except Exception as e:
    logger.error("Failed to import required package `mtk_converter`.")
    logger.error(
        "Please install the Neuropilot SDK bundled `mtk_converter` wheel for the current Python environment."
    )
    logger.error(
        "Look for the package archive under the SDK, for example: "
        "`<sdk_root>/offline_tool/mtk_converter_*_packages.zip`."
    )
    logger.error(f"Current Python: {sys.executable} ({sys.version.split()[0]})")
    logger.error("Install the wheel that matches this Python version and platform, then run again.")
    logger.error(f"Original import error: {e}")
    raise


TARGET = "host"
DEFAULT_EP_OPSET_VERSION = 13
DEFAULT_EP_IR_VERSION = 7
MICROSOFT_OPSET_VERSION = 1


SOC_TO_MDLA_ARCH = {
    "mt6993": "mdla6.1",
    "mt6991": "mdla5.5,mvpu2.5",
}


def create_ort_session(model_path):
    # Set thread counts explicitly so ORT does not try to pin worker threads.
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    return ort.InferenceSession(model_path, sess_options=session_options)


def print_pipeline_summary(steps):
    logger.info("")
    logger.info("================== Pipeline Summary ==================")
    logger.info(f"{'Step':<20} {'Status':<10}")
    logger.info("-" * 30)
    for name, status in steps:
        status_str = "✅" if status else "❌" if status is False else ""
        logger.info(f"{name:<20} {status_str:<10}")


def create_epcontext_node(input_names, output_names, dla_filename):
    return helper.make_node(
        'EPContext',
        inputs=input_names,
        outputs=output_names,
        name='NeuronpilotContext',
        domain='com.microsoft',
        embed_mode=0,
        ep_cache_context=dla_filename,
        source='Neuropilot'
    )


def build_epcontext_model(
    input_model_path,
    dla_filename,
    output_model_path,
    opset_version=DEFAULT_EP_OPSET_VERSION,
    ir_version=DEFAULT_EP_IR_VERSION,
):
    model = onnx.load(input_model_path)

    input_names = [inp.name for inp in model.graph.input]
    output_names = [out.name for out in model.graph.output]

    epcontext_node = create_epcontext_node(input_names, output_names, dla_filename)

    graph = helper.make_graph(
        [epcontext_node],
        'epcontext_graph',
        list(model.graph.input),
        list(model.graph.output)
    )

    new_model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", opset_version),
            helper.make_operatorsetid("com.microsoft", MICROSOFT_OPSET_VERSION),
        ],
        producer_name='epcontext_generator'
    )

    new_model.ir_version = ir_version

    onnx.save(new_model, output_model_path)

    logger.info("")
    logger.info("[EPContext Model]")
    logger.info(f"  Saved to: {output_model_path}")
    logger.info(f"  IR version: {new_model.ir_version}")
    opset_info = [f'{op.domain or "ai.onnx"}:{op.version}' for op in new_model.opset_import]
    logger.info(f"  Opset imports: {opset_info}")
    logger.info("  Inputs:")
    for inp in model.graph.input:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type)
        logger.info(f"    {inp.name}: shape={shape}, dtype={dtype}")
    logger.info("  Outputs:")
    for outp in model.graph.output:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in outp.type.tensor_type.shape.dim]
        dtype = onnx.TensorProto.DataType.Name(outp.type.tensor_type.elem_type)
        logger.info(f"    {outp.name}: shape={shape}, dtype={dtype}")


def run_output_check(original_onnx, rewritten_onnx, data_dir):
    model_orig = onnx.load(original_onnx)
    model_rewritten = onnx.load(rewritten_onnx)

    inputs_orig = [inp.name for inp in model_orig.graph.input]

    input_data = {}
    for inp in model_orig.graph.input:
        shape = [d.dim_value if d.dim_value else 1 for d in inp.type.tensor_type.shape.dim]
        dtype_map = {
            1: np.float32, 2: np.uint8, 3: np.int8, 6: np.int32,
            7: np.int64, 9: np.bool_, 10: np.float16, 11: np.double
        }
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)
        input_data[inp.name] = np.random.randn(*shape).astype(dtype)
        np.save(os.path.join(data_dir, f"{inp.name}.npy"), input_data[inp.name])

    sess_orig = create_ort_session(original_onnx)
    output_orig = sess_orig.run(None, {name: input_data[name] for name in inputs_orig})

    rewritten_input_data = {}
    for i, inp in enumerate(model_rewritten.graph.input):
        orig_name = inputs_orig[i] if i < len(inputs_orig) else inputs_orig[-1]
        if inp.name.endswith('_nhwc'):
            data = input_data[orig_name]
            if len(data.shape) == 4:
                data = np.transpose(data, [0, 2, 3, 1])
            rewritten_input_data[inp.name] = data
        else:
            rewritten_input_data[inp.name] = input_data.get(inp.name, input_data[orig_name])

    sess_rewritten = create_ort_session(rewritten_onnx)
    output_rewritten = sess_rewritten.run(None, rewritten_input_data)

    outputs_orig = [out.name for out in model_orig.graph.output]
    outputs_rewritten = [out.name for out in model_rewritten.graph.output]

    if len(outputs_orig) != len(outputs_rewritten):
        raise RuntimeError(
            "Output check failed: output count mismatch "
            f"(original={len(outputs_orig)}, rewritten={len(outputs_rewritten)})"
        )

    if outputs_orig != outputs_rewritten:
        raise RuntimeError(
            "Output check failed: output names mismatch "
            f"(original={outputs_orig}, rewritten={outputs_rewritten})"
        )

    for i, (orig_name, rewritten_name) in enumerate(zip(outputs_orig, outputs_rewritten)):
        orig_out = output_orig[i]
        rewritten_out = output_rewritten[i]

        if rewritten_name.endswith('_nhwc') and len(orig_out.shape) == 4:
            rewritten_out = np.transpose(rewritten_out, [0, 3, 1, 2])

        np.save(os.path.join(data_dir, f"{orig_name}.npy"), orig_out)

        if not np.allclose(orig_out, rewritten_out, rtol=1e-3, atol=1e-5):
            raise RuntimeError(f"Output check failed for output {orig_name}")

    logger.info(f"[Output check] Passed, data saved to {data_dir}")


def _copy_dim(dim):
    new_dim = onnx.TensorShapeProto.Dimension()
    if dim.HasField("dim_value"):
        new_dim.dim_value = dim.dim_value
    elif dim.HasField("dim_param"):
        new_dim.dim_param = dim.dim_param
    return new_dim


def _dims_to_str(dims):
    out = []
    for d in dims:
        if d.HasField("dim_value"):
            out.append(str(d.dim_value))
        elif d.HasField("dim_param"):
            out.append(d.dim_param)
        else:
            out.append("?")
    return "x".join(out)


def normalize_soc_list(raw_socs):
    socs = []
    for item in raw_socs:
        for soc in item.split(","):
            soc = soc.strip()
            if soc:
                socs.append(soc)
    return socs


def rewrite_onnx_nhwc_io(input_onnx_path, output_onnx_path):
    model = onnx.load(input_onnx_path)

    graph = model.graph

    input_transpose_nodes = []
    output_transpose_nodes = []
    new_inputs = []
    new_outputs = []

    input_cnt = 0
    output_cnt = 0

    logger.info("========== Rewrite ONNX NHWC IO ==========")

    for input_tensor in graph.input:
        name = input_tensor.name
        ttype = input_tensor.type.tensor_type

        if ttype.HasField("shape") and len(ttype.shape.dim) == 4:
            input_cnt += 1
            nhwc_name = name + "_nhwc"
            dims = ttype.shape.dim

            nhwc_dims = [
                _copy_dim(dims[0]),
                _copy_dim(dims[2]),
                _copy_dim(dims[3]),
                _copy_dim(dims[1]),
            ]

            logger.info(f"[Input ] {name}: NCHW {_dims_to_str(dims)} -> NHWC {_dims_to_str(nhwc_dims)}")
            logger.info(f"         + Insert Transpose {name}_nhwc_to_nchw perm=[0,3,1,2]")

            new_input = helper.make_tensor_value_info(nhwc_name, ttype.elem_type, None)
            new_input.type.tensor_type.shape.dim.extend(nhwc_dims)
            new_inputs.append(new_input)

            input_transpose_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[nhwc_name],
                    outputs=[name],
                    perm=[0, 3, 1, 2],
                    name=name + "_nhwc_to_nchw"
                )
            )
        else:
            new_inputs.append(deepcopy(input_tensor))

    for output_tensor in graph.output:
        name = output_tensor.name
        ttype = output_tensor.type.tensor_type

        if ttype.HasField("shape") and len(ttype.shape.dim) == 4:
            output_cnt += 1
            nhwc_name = name + "_nhwc"
            dims = ttype.shape.dim

            nhwc_dims = [
                _copy_dim(dims[0]),
                _copy_dim(dims[2]),
                _copy_dim(dims[3]),
                _copy_dim(dims[1]),
            ]

            logger.info(f"[Output] {name}: NCHW {_dims_to_str(dims)} -> NHWC {_dims_to_str(nhwc_dims)}")
            logger.info(f"         + Insert Transpose {name}_nchw_to_nhwc perm=[0,2,3,1]")

            new_output = helper.make_tensor_value_info(nhwc_name, ttype.elem_type, None)
            new_output.type.tensor_type.shape.dim.extend(nhwc_dims)
            new_outputs.append(new_output)

            output_transpose_nodes.append(
                helper.make_node(
                    "Transpose",
                    inputs=[name],
                    outputs=[nhwc_name],
                    perm=[0, 2, 3, 1],
                    name=name + "_nchw_to_nhwc"
                )
            )
        else:
            new_outputs.append(deepcopy(output_tensor))

    modified = (input_cnt > 0) or (output_cnt > 0)

    if modified:
        del graph.input[:]
        graph.input.extend(new_inputs)

        del graph.output[:]
        graph.output.extend(new_outputs)

        original_nodes = list(graph.node)
        del graph.node[:]

        graph.node.extend(input_transpose_nodes)
        graph.node.extend(original_nodes)
        graph.node.extend(output_transpose_nodes)

        model = shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, output_onnx_path)

        logger.info("")
        logger.info("[Summary]")
        logger.info(f"  Inputs processed : {input_cnt}")
        logger.info(f"  Outputs processed: {output_cnt}")
        logger.info(f"  New input nodes  : {len(input_transpose_nodes)}")
        logger.info(f"  New output nodes : {len(output_transpose_nodes)}")
        logger.info(f"[OK] Saved to {output_onnx_path}")
    else:
        logger.info("[NHWC Rewrite] No 4D IO found, skip rewrite")

    return modified


def convert_for_soc(
    model_name: str,
    original_onnx: str,
    rewritten_onnx: str,
    soc: str,
    sdk_path: str,
    intermediate_root_dir: str,
    converted_root_dir: str,
    ep_opset_version: int,
    ep_ir_version: int,
):
    ncc_bin = os.path.join(sdk_path, TARGET, "bin", "ncc-tflite")
    steps = [
        ("ONNX→TFLite", None),
        ("TFLite→DLA", None),
        ("EPContext wrap", None),
    ]
    dla_path = ""
    epcontext_onnx_path = ""

    try:
        converted_dir = os.path.join(converted_root_dir, soc)
        tflite_dir = os.path.join(intermediate_root_dir, "tflite", soc)
        os.makedirs(converted_dir, exist_ok=True)
        os.makedirs(tflite_dir, exist_ok=True)

        try:
            tflite_path = os.path.join(tflite_dir, f"{model_name}.tflite")
            cmd = ["mtk_onnx_converter", "--input_model_file", rewritten_onnx, "--output_file", tflite_path]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                steps[0] = (steps[0][0], False)
                raise RuntimeError(f"mtk_onnx_converter failed: {r.stderr}")
            steps[0] = (steps[0][0], True)
        except Exception as e:
            steps[0] = (steps[0][0], False)
            raise

        try:
            mdla_arch = SOC_TO_MDLA_ARCH[soc]
            dla_path = os.path.join(converted_dir, f"{model_name}.dla")
            cmd = [
                ncc_bin,
                "--arch", mdla_arch,
                "--l1-size-kb", "7168",
                "--num-mdla", "4",
                "--opt-bw",
                "--relax-fp32",
                "--show-memory-summary",
                "--show-exec-plan",
                tflite_path,
                "-d", dla_path
            ]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.returncode != 0:
                steps[1] = (steps[1][0], False)
                raise RuntimeError(f"ncc-tflite failed: {r.stderr}")
            steps[1] = (steps[1][0], True)
        except Exception as e:
            steps[1] = (steps[1][0], False)
            raise

        try:
            dla_filename = os.path.basename(dla_path)
            epcontext_onnx_path = os.path.join(converted_dir, f"{model_name}_epcontext.onnx")
            build_epcontext_model(
                original_onnx,
                dla_filename,
                epcontext_onnx_path,
                opset_version=ep_opset_version,
                ir_version=ep_ir_version,
            )
            steps[2] = (steps[2][0], True)
        except Exception as e:
            steps[2] = (steps[2][0], False)
            raise RuntimeError(f"EPContext wrap failed: {e}")

        return {
            "model": model_name,
            "soc": soc,
            "converted_dir": converted_dir,
            "tflite_path": tflite_path,
            "dla_path": dla_path,
            "epcontext_onnx_path": epcontext_onnx_path,
        }

    except Exception as e:
        logger.error("")
        logger.error(f"[Pipeline error] {e}")
        raise

    finally:
        print_pipeline_summary(steps)
        if dla_path:
            logger.info(f"DLA output path: {dla_path}")
        if epcontext_onnx_path:
            logger.info(f"EPContext ONNX path: {epcontext_onnx_path}")


def convert_and_wrap(
    model_path: str,
    socs,
    sdk_path: str,
    output_dir: str,
    ep_opset_version: int = DEFAULT_EP_OPSET_VERSION,
    ep_ir_version: int = DEFAULT_EP_IR_VERSION,
):
    if not socs:
        raise ValueError("At least one target SoC must be provided after normalization.")

    model_name, ext = os.path.splitext(os.path.basename(model_path))
    if ext.lower() != ".onnx":
        raise ValueError("Only ONNX models are supported")

    intermediate_root = os.path.join(output_dir, "intermediate")
    converted_root = os.path.join(output_dir, "converted")
    onnx_dir = os.path.join(intermediate_root, "onnx")
    data_dir = os.path.join(intermediate_root, "output_check")
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(converted_root, exist_ok=True)

    original_onnx = os.path.join(onnx_dir, os.path.basename(model_path))
    shutil.copy2(model_path, original_onnx)

    try:
        rewritten_onnx = os.path.join(onnx_dir, f"{model_name}_nhwc.onnx")
        modified = rewrite_onnx_nhwc_io(original_onnx, rewritten_onnx)
        if not modified:
            rewritten_onnx = original_onnx
    except Exception as e:
        print_pipeline_summary([("Rewrite NHWC", False)])
        raise RuntimeError(f"Rewrite NHWC failed: {e}")

    try:
        run_output_check(original_onnx, rewritten_onnx, data_dir)
    except Exception as e:
        print_pipeline_summary([("Rewrite NHWC", True), ("Output check", False)])
        raise RuntimeError(f"Output check failed: {e}")

    logger.info("")
    logger.info("[Output Layout]")
    logger.info(f"  Intermediate root: {intermediate_root}")
    logger.info(f"  Converted root   : {converted_root}")
    logger.info(f"  Target SoCs      : {', '.join(socs)}")
    logger.info(f"  EP opset version : {ep_opset_version}")
    logger.info(f"  EP IR version    : {ep_ir_version}")

    results = []
    for soc in socs:
        results.append(
            convert_for_soc(
                model_name=model_name,
                original_onnx=original_onnx,
                rewritten_onnx=rewritten_onnx,
                soc=soc,
                sdk_path=sdk_path,
                intermediate_root_dir=intermediate_root,
                converted_root_dir=converted_root,
                ep_opset_version=ep_opset_version,
                ep_ir_version=ep_ir_version,
            )
        )

    logger.info(f"Output check data directory: {data_dir}")
    return {
        "model": model_name,
        "intermediate_root": intermediate_root,
        "converted_root": converted_root,
        "results": results,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="ONNX → TFLite → DLA pipeline with EPContext wrapper generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to input ONNX model"
    )
    parser.add_argument(
        "--soc",
        nargs="+",
        required=True,
        help="Target SoC list, e.g. --soc mt6991 mt6993 or --soc mt6991,mt6993"
    )
    parser.add_argument(
        "--sdk",
        type=str,
        required=True,
        help="Path to Neuropilot SDK directory (neuron_sdk)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--ep-opset-version",
        type=int,
        default=DEFAULT_EP_OPSET_VERSION,
        help=(
            f"Opset version used for generated EPContext ONNX "
            f"(default: {DEFAULT_EP_OPSET_VERSION}, must be >= {DEFAULT_EP_OPSET_VERSION})"
        )
    )
    parser.add_argument(
        "--ep-ir-version",
        type=int,
        default=DEFAULT_EP_IR_VERSION,
        help=(
            f"IR version used for generated EPContext ONNX "
            f"(default: {DEFAULT_EP_IR_VERSION}, must be >= {DEFAULT_EP_IR_VERSION})"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    socs = normalize_soc_list(args.soc)
    unsupported_socs = [soc for soc in socs if soc not in SOC_TO_MDLA_ARCH]
    if unsupported_socs:
        raise ValueError(
            f"Unsupported SoC(s): {unsupported_socs}. Supported values: {sorted(SOC_TO_MDLA_ARCH)}"
        )
    if args.ep_opset_version < DEFAULT_EP_OPSET_VERSION:
        raise ValueError(
            f"--ep-opset-version must be >= {DEFAULT_EP_OPSET_VERSION}, got {args.ep_opset_version}"
        )
    if args.ep_ir_version < DEFAULT_EP_IR_VERSION:
        raise ValueError(
            f"--ep-ir-version must be >= {DEFAULT_EP_IR_VERSION}, got {args.ep_ir_version}"
        )
    convert_and_wrap(
        model_path=args.model,
        socs=socs,
        sdk_path=args.sdk,
        output_dir=args.output,
        ep_opset_version=args.ep_opset_version,
        ep_ir_version=args.ep_ir_version,
    )
