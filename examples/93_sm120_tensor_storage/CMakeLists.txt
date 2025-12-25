# Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_NVCC_ARCHS MATCHES "120a|121a")
cutlass_example_add_executable(
  93_sm120_tensor_storage_probe
  93_sm120_tensor_storage_probe.cu
)
endif()
