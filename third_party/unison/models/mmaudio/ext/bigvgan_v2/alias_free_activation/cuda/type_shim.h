// Vendored from UNISON and modified for CrossFeed.
// https://github.com/lizhaoqing/UNISON/blob/7608c183f9f76e8da73261c95e923376b22f1e33/unison/models/mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/type_shim.h

/* coding=utf-8
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/ATen.h>
#include "compat.h"

#define DISPATCH_FLOAT_HALF_AND_BFLOAT(TYPE, NAME, ...)                 \
	switch (TYPE)                                                       \
	{                                                                   \
	case at::ScalarType::Float:                                         \
	{                                                                   \
		using scalar_t = float;                                         \
		__VA_ARGS__;                                                    \
		break;                                                          \
	}                                                                   \
	case at::ScalarType::Half:                                          \
	{                                                                   \
		using scalar_t = at::Half;                                      \
		__VA_ARGS__;                                                    \
		break;                                                          \
	}                                                                   \
	case at::ScalarType::BFloat16:                                      \
	{                                                                   \
		using scalar_t = at::BFloat16;                                  \
		__VA_ARGS__;                                                    \
		break;                                                          \
	}                                                                   \
	default:                                                            \
		AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
	}

#define DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(TYPEIN, TYPEOUT, NAME, ...) \
	switch (TYPEIN)                                                            \
	{                                                                          \
	case at::ScalarType::Float:                                                \
	{                                                                          \
		using scalar_t_in = float;                                             \
		switch (TYPEOUT)                                                       \
		{                                                                      \
		case at::ScalarType::Float:                                            \
		{                                                                      \
			using scalar_t_out = float;                                        \
			__VA_ARGS__;                                                       \
			break;                                                             \
		}                                                                      \
		case at::ScalarType::Half:                                             \
		{                                                                      \
			using scalar_t_out = at::Half;                                     \
			__VA_ARGS__;                                                       \
			break;                                                             \
		}                                                                      \
		case at::ScalarType::BFloat16:                                         \
		{                                                                      \
			using scalar_t_out = at::BFloat16;                                 \
			__VA_ARGS__;                                                       \
			break;                                                             \
		}                                                                      \
		default:                                                               \
			AT_ERROR(#NAME, " not implemented for '", toString(TYPEOUT), "'"); \
		}                                                                      \
		break;                                                                 \
	}                                                                          \
	case at::ScalarType::Half:                                                 \
	{                                                                          \
		using scalar_t_in = at::Half;                                          \
		using scalar_t_out = at::Half;                                         \
		__VA_ARGS__;                                                           \
		break;                                                                 \
	}                                                                          \
	case at::ScalarType::BFloat16:                                             \
	{                                                                          \
		using scalar_t_in = at::BFloat16;                                      \
		using scalar_t_out = at::BFloat16;                                     \
		__VA_ARGS__;                                                           \
		break;                                                                 \
	}                                                                          \
	default:                                                                   \
		AT_ERROR(#NAME, " not implemented for '", toString(TYPEIN), "'");      \
	}
