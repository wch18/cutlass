/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
 *  Probe the storage layout for FP8/FP6/FP4 matrix multiply inputs on SM120A GPUs.
 *
 *  The example launches a small kernel that prints the contents of Tensors A, B, and C
 *  from global memory, the shared memory staging buffers, and the registers used by
 *  one thread. MX formats show both their packed data and scale factors so it is easy
 *  to see how block scaled tensors are materialized during the mainloop.
 *
 *  Usage:
 *    $ ./examples/93_sm120_tensor_storage/93_sm120_tensor_storage_probe
 */

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <type_traits>
#include <mma.h>

#include "cutlass/cutlass.h"
#include "cutlass/float8.h"
#include "cutlass/float_subbyte.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

using cutlass::NumericConverter;

// We size the problem to fit a single WMMA tile so the mainloop can run on
// Tensor Cores. The tile is still small enough that printing every element is
// practical.
constexpr int kRowsA = 16;
constexpr int kColsA = 16;
constexpr int kRowsB = kColsA;
constexpr int kColsB = 16;
constexpr int kRowsC = kRowsA;
constexpr int kColsC = kColsB;

// Simple helper to print both the raw byte that is stored in memory and the
// floating point value obtained after conversion.
template <typename Element>
CUTLASS_DEVICE void print_element_bits(const char* label, Element value) {
  NumericConverter<float, Element> convert;
  auto as_float = convert(value);

  uint8_t raw = 0;
  static_assert(sizeof(Element) <= sizeof(raw), "expected sub-byte storage");
  memcpy(&raw, &value, sizeof(Element));
  printf("    %-12s : 0x%02x -> %0.4f\n", label, raw, as_float);
}

template <typename Element>
CUTLASS_DEVICE float scaled_value(Element value) {
  NumericConverter<float, Element> convert;
  return convert(value);
}

template <typename Element, typename Scale>
CUTLASS_DEVICE float scaled_value(Element value, Scale scale) {
  NumericConverter<float, Element> convert_data;
  NumericConverter<float, Scale> convert_scale;
  return convert_data(value) * convert_scale(scale);
}

// A small structure describing the tensor format for one operand.
template <typename DataType, typename ScaleType, bool HasScale>
struct OperandFormat {
  using Data = DataType;
  using Scale = typename std::conditional<HasScale, ScaleType, cutlass::half_t>::type;
  static constexpr bool kHasScale = HasScale;
  static CUTLASS_HOST_DEVICE const char* name() { return ""; }
};

struct NonMxFp8 : OperandFormat<cutlass::float_e4m3_t, cutlass::half_t, false> {
  static CUTLASS_HOST_DEVICE const char* name() { return "FP8"; }
};

struct MxFp8 : OperandFormat<cutlass::mx_float8_t<cutlass::float_e4m3_t>::DataType,
                              cutlass::mx_float8_t<cutlass::float_e4m3_t>::ScaleFactorType,
                              true> {
  static CUTLASS_HOST_DEVICE const char* name() { return "MXFP8"; }
};

struct NonMxFp6 : OperandFormat<cutlass::float_e3m2_t, cutlass::half_t, false> {
  static CUTLASS_HOST_DEVICE const char* name() { return "FP6"; }
};

struct MxFp6 : OperandFormat<cutlass::mx_float6_t<cutlass::float_e3m2_t>::DataType,
                              cutlass::mx_float6_t<cutlass::float_e3m2_t>::ScaleFactorType,
                              true> {
  static CUTLASS_HOST_DEVICE const char* name() { return "MXFP6"; }
};

struct NonMxFp4 : OperandFormat<cutlass::float_e2m1_t, cutlass::half_t, false> {
  static CUTLASS_HOST_DEVICE const char* name() { return "FP4"; }
};

struct MxFp4 : OperandFormat<cutlass::mx_float4_t<cutlass::float_e2m1_t>::DataType,
                              cutlass::mx_float4_t<cutlass::float_e2m1_t>::ScaleFactorType,
                              true> {
  static CUTLASS_HOST_DEVICE const char* name() { return "MXFP4"; }
};

// The device kernel performs three steps:
// 1. print global memory values for the matrix tile each thread block owns.
// 2. stage operands through shared memory and print their layout.
// 3. move the operands into registers, compute a small GEMM, and print register contents.
template <typename FormatA, typename FormatB>
__global__ void probe_kernel(typename FormatA::Data const* A, typename FormatA::Scale const* scaleA,
                             typename FormatB::Data const* B, typename FormatB::Scale const* scaleB,
                             float* C_global, int lda, int ldb, int ldc, const char* tag) {
  __shared__ typename FormatA::Data shared_A[kRowsA * kColsA];
  __shared__ typename FormatB::Data shared_B[kRowsB * kColsB];
  __shared__ float shared_C[kRowsC * kColsC];

  int tid = threadIdx.x;
  if (blockIdx.x != 0) {
    return;
  }

  // Only lane 0 emits printf, but all lanes participate in the Tensor Core
  // MMA instructions.
  if (tid == 0) {
    printf("\n===== %s (%s x %s) =====\n", tag, FormatA::name(), FormatB::name());
  }

  auto print_global = [&](auto ptr, auto scale_ptr, int rows, int cols, const char* name) {
    printf("Global %s (row-major)\n", name);
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        int idx = r * cols + c;
        auto value = ptr[idx];
        if constexpr (FormatA::kHasScale || FormatB::kHasScale) {
          if (scale_ptr) {
            auto scaled = scaled_value(value, scale_ptr[idx]);
            print_element_bits("byte", value);
            using ScalePtrT = typename std::remove_pointer<decltype(scale_ptr)>::type;
            using ScaleT = typename std::remove_const<ScalePtrT>::type;
            NumericConverter<float, ScaleT> convert_sf;
            auto scale_val = scale_ptr[idx];
            print_element_bits("scale", scale_val);
            printf("      scale(float): %0.4f\n", convert_sf(scale_val));
            printf("      scaled elem : %0.4f\n", scaled);
          } else {
            print_element_bits("byte", value);
          }
        } else {
          print_element_bits("byte", value);
        }
      }
    }
  };

  if (tid == 0) {
    print_global(A, scaleA, kRowsA, kColsA, "A");
    print_global(B, scaleB, kRowsB, kColsB, "B");
  }

  // Stage operands through shared memory.
  // Cooperative staging so the entire warp participates in the subsequent
  // Tensor Core operation.
  for (int linear = tid; linear < kRowsA * kColsA; linear += blockDim.x) {
    shared_A[linear] = A[linear];
  }
  for (int linear = tid; linear < kRowsB * kColsB; linear += blockDim.x) {
    shared_B[linear] = B[linear];
  }
  __syncthreads();

  if (tid == 0) {
    printf("Shared A\n");
    for (int r = 0; r < kRowsA; ++r) {
      for (int c = 0; c < kColsA; ++c) {
        int idx = r * kColsA + c;
        print_element_bits("byte", shared_A[idx]);
      }
    }

    printf("Shared B\n");
    for (int r = 0; r < kRowsB; ++r) {
      for (int c = 0; c < kColsB; ++c) {
        int idx = r * kColsB + c;
        print_element_bits("byte", shared_B[idx]);
      }
    }
  }

  // Load registers and compute a tiny GEMM using Tensor Core MMA. We convert
  // the scaled operands to half so the WMMA API can issue a single 16x16x16
  // operation on Tensor Cores. The input storage remains FP8/FP6/FP4 (MX or
  // non-MX) and is what gets printed above.
  __shared__ cutlass::half_t shared_A_converted[kRowsA * kColsA];
  __shared__ cutlass::half_t shared_B_converted[kRowsB * kColsB];

  for (int linear = tid; linear < kRowsA * kColsA; linear += blockDim.x) {
    float val = FormatA::kHasScale ? scaled_value(shared_A[linear], scaleA ? scaleA[linear] : typename FormatA::Scale{})
                                   : scaled_value(shared_A[linear]);
    shared_A_converted[linear] = cutlass::half_t(val);
  }
  for (int linear = tid; linear < kRowsB * kColsB; linear += blockDim.x) {
    float val = FormatB::kHasScale ? scaled_value(shared_B[linear], scaleB ? scaleB[linear] : typename FormatB::Scale{})
                                   : scaled_value(shared_B[linear]);
    shared_B_converted[linear] = cutlass::half_t(val);
  }
  __syncthreads();

  using namespace nvcuda;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fragA;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fragB;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fragC;

  wmma::fill_fragment(fragC, 0.0f);
  wmma::load_matrix_sync(fragA, reinterpret_cast<half const*>(shared_A_converted), kColsA);
  wmma::load_matrix_sync(fragB, reinterpret_cast<half const*>(shared_B_converted), kColsB);
  wmma::mma_sync(fragC, fragA, fragB, fragC);
  wmma::store_matrix_sync(shared_C, fragC, kColsC, wmma::mem_row_major);
  __syncthreads();

  // Broadcast accumulators from shared memory and store to global. Only one
  // lane prints to avoid flooding stdout, but all lanes execute the MMA above.
  for (int linear = tid; linear < kRowsC * kColsC; linear += blockDim.x) {
    C_global[linear] = shared_C[linear];
  }
  __syncthreads();

  if (tid == 0) {
    printf("Register/fragment C (Tensor Core accumulators)\n");
    for (int i = 0; i < fragC.num_elements; ++i) {
      printf("    fragC[%02d]   : %0.4f\n", i, fragC.x[i]);
    }

    printf("Shared C (float accumulators)\n");
    for (int i = 0; i < kRowsC * kColsC; ++i) {
      printf("    s[%02d]        : %0.4f\n", i, shared_C[i]);
    }

    printf("Global C\n");
    for (int r = 0; r < kRowsC; ++r) {
      for (int c = 0; c < kColsC; ++c) {
        int idx = r * ldc + c;
        printf("    C(%d,%d) = %0.4f\n", r, c, C_global[idx]);
      }
    }
  }
}

// Host helper that fills tensors and launches the probe kernel.
template <typename FormatA, typename FormatB>
bool run_probe(const char* tag) {
  using DataA = typename FormatA::Data;
  using DataB = typename FormatB::Data;

  cutlass::HostTensor<DataA, cutlass::layout::RowMajor> A({kRowsA, kColsA});
  cutlass::HostTensor<DataB, cutlass::layout::RowMajor> B({kRowsB, kColsB});
  cutlass::HostTensor<float, cutlass::layout::RowMajor> C({kRowsC, kColsC});

  for (int i = 0; i < kRowsA * kColsA; ++i) {
    A.host_data()[i] = DataA((i % 7) + 1);
  }
  for (int i = 0; i < kRowsB * kColsB; ++i) {
    B.host_data()[i] = DataB((i % 5) + 1);
  }

  typename FormatA::Scale* scaleA_ptr = nullptr;
  typename FormatB::Scale* scaleB_ptr = nullptr;

  cutlass::HostTensor<typename FormatA::Scale, cutlass::layout::PackedVectorLayout> scaleA;
  cutlass::HostTensor<typename FormatB::Scale, cutlass::layout::PackedVectorLayout> scaleB;

  if constexpr (FormatA::kHasScale) {
    scaleA.reset({kRowsA * kColsA});
    for (int i = 0; i < kRowsA * kColsA; ++i) {
      scaleA.host_data()[i] = typename FormatA::Scale(0.5f + 0.1f * i);
    }
    scaleA_ptr = scaleA.device_data();
  }

  if constexpr (FormatB::kHasScale) {
    scaleB.reset({kRowsB * kColsB});
    for (int i = 0; i < kRowsB * kColsB; ++i) {
      scaleB.host_data()[i] = typename FormatB::Scale(0.25f + 0.05f * i);
    }
    scaleB_ptr = scaleB.device_data();
  }

  A.sync_device();
  B.sync_device();
  if constexpr (FormatA::kHasScale) {
    scaleA.sync_device();
  }
  if constexpr (FormatB::kHasScale) {
    scaleB.sync_device();
  }

  dim3 grid(1);
  dim3 block(32);
  probe_kernel<FormatA, FormatB><<<grid, block>>>(A.device_data(), scaleA_ptr, B.device_data(), scaleB_ptr,
                                                  C.device_data(), A.layout().stride(0), B.layout().stride(0),
                                                  C.layout().stride(0), tag);
  cudaError_t status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << std::endl;
    return false;
  }

  C.sync_host();
  std::cout << "Host view of C after probe:\n" << C.host_view() << "\n";
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  cutlass::CommandLine cmd(argc, argv);
  bool mx_only = cmd.check_cmd_line_flag("mx_only");
  bool plain_only = cmd.check_cmd_line_flag("plain_only");

  bool ok = true;

  if (!plain_only) {
    ok &= run_probe<MxFp8, MxFp8>("MXFP8 x MXFP8");
    ok &= run_probe<MxFp6, MxFp6>("MXFP6 x MXFP6");
    ok &= run_probe<MxFp4, MxFp4>("MXFP4 x MXFP4");
  }

  if (!mx_only) {
    ok &= run_probe<NonMxFp8, NonMxFp8>("FP8 x FP8");
    ok &= run_probe<NonMxFp6, NonMxFp6>("FP6 x FP6");
    ok &= run_probe<NonMxFp4, NonMxFp4>("FP4 x FP4");
  }

  return ok ? 0 : -1;
}

#else

int main(int, char**) {
  std::cout << "SM120A support is required to run this example." << std::endl;
  return 0;
}

#endif
