//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#import "../osd/mtlPatchTable.h"
#import <Metal/Metal.h>
#import "../far/patchTable.h"
#import "../osd/cpuPatchTable.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
namespace Osd {

MTLPatchTable::MTLPatchTable()
:
_indexBuffer(nil),
_patchParamBuffer(nil)
{

}


MTLPatchTable::~MTLPatchTable()
{
    
}

static id<MTLBuffer> createBuffer(const void* data, const size_t length,
                                  MTLContext* context)
{
#if TARGET_OS_EMBEDDED
    return [context->device newBufferWithBytes:data length:length options:MTLResourceOptionCPUCacheModeDefault];
#else
  @autoreleasepool {
    auto cmdBuf = [context->commandQueue commandBuffer];
    auto blitEncoder = [cmdBuf blitCommandEncoder];

    auto stageBuffer = [context->device newBufferWithBytes:data length:length options:MTLResourceOptionCPUCacheModeDefault];

    auto finalBuffer = [context->device newBufferWithLength:length options:MTLResourceStorageModePrivate];

    [blitEncoder copyFromBuffer:stageBuffer sourceOffset:0 toBuffer:finalBuffer destinationOffset:0 size:length];
    [blitEncoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return finalBuffer;
  }
#endif
}


MTLPatchTable* MTLPatchTable::Create(const Far::PatchTable *farPatchTable, MTLContext* context)
{
    auto patchTable = new MTLPatchTable();
    if(patchTable->allocate(farPatchTable, context))
        return patchTable;

    delete patchTable;
    assert(0 && "MTLPatchTable Creation Failed");
    return nullptr;
}

bool MTLPatchTable::allocate(Far::PatchTable const *farPatchTable, MTLContext* context)
{
    CpuPatchTable cpuTable(farPatchTable);

    auto numPatchArrays = cpuTable.GetNumPatchArrays();
    auto indexSize = cpuTable.GetPatchIndexSize();
    auto patchParamSize = cpuTable.GetPatchParamSize();

    _patchArrays.assign(cpuTable.GetPatchArrayBuffer(), cpuTable.GetPatchArrayBuffer() + numPatchArrays);
    
    _indexBuffer = createBuffer(cpuTable.GetPatchIndexBuffer(), indexSize * sizeof(unsigned), context);
    if(_indexBuffer == nil)
        return false;

    _indexBuffer.label = @"OSD PatchIndexBuffer";

    _patchParamBuffer = createBuffer(cpuTable.GetPatchParamBuffer(), patchParamSize * sizeof(PatchParam), context);
    if(_patchParamBuffer == nil)
        return false;

    _patchParamBuffer.label = @"OSD PatchParamBuffer";

    return true;
}

} //end namespace Osd
} //end namespace OPENSUBDIV_VERSION
} //end namespace OpenSubdiv
