/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/runtime/common.h"

#include <optional>

namespace tensorrt_llm::batch_manager
{

class TrtGptModelOptionalParams
{
    using KvCacheConfig = kv_cache_manager::KvCacheConfig;

public:
    using SizeType = tensorrt_llm::runtime::SizeType;

    explicit TrtGptModelOptionalParams(KvCacheConfig const& kvCacheConfig = KvCacheConfig{},
        std::optional<SizeType> maxNumSequences = std::nullopt, bool enableTrtOverlap = true,
        bool useContextFMHAForGeneration = false)
        : kvCacheConfig{kvCacheConfig}
        , maxNumSequences{maxNumSequences}
        , enableTrtOverlap{enableTrtOverlap}
        , useContextFMHAForGeneration(useContextFMHAForGeneration)
    {
    }

    KvCacheConfig kvCacheConfig;
    std::optional<SizeType> maxNumSequences;
    bool enableTrtOverlap;
    bool useContextFMHAForGeneration;
};

} // namespace tensorrt_llm::batch_manager
