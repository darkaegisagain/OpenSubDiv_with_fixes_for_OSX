//
//   Copyright 2015 Pixar
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

#import "mtlViewer.h"

#import <simd/simd.h>
#import <algorithm>
#import <cfloat>
#import <fstream>
#import <iostream>
#import <iterator>
#import <string>
#import <sstream>
#import <vector>
#import <memory>

#import <far/error.h>
#import <osd/mesh.h>
#import <osd/cpuVertexBuffer.h>
#import <osd/cpuEvaluator.h>
#import <osd/cpuPatchTable.h>
#import <osd/mtlLegacyGregoryPatchTable.h>
#import <osd/mtlVertexBuffer.h>
#import <osd/mtlMesh.h>
#import <osd/mtlPatchTable.h>
#import <osd/mtlComputeEvaluator.h>
#import <osd/mtlPatchShaderSource.h>

#import "../common/simple_math.h"
#import "../../regression/common/far_utils.h"
#import "init_shapes.h"
#import "../common/mtlUtils.h"
#import "../common/mtlControlMeshDisplay.h"

#define VERTEX_BUFFER_INDEX 0
#define PATCH_INDICES_BUFFER_INDEX 1
#define CONTROL_INDICES_BUFFER_INDEX 2
#define OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX 3
#define OSD_PATCHPARAM_BUFFER_INDEX 4
#define OSD_VALENCE_BUFFER_INDEX 6
#define OSD_QUADOFFSET_BUFFER_INDEX 7
#define OSD_PERPATCHTESSFACTORS_BUFFER_INDEX 8
#define HULL_BUFFER_INDEX 9
#define QUAD_TESSFACTORS_INDEX 10
#define OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX
#define OSD_PATCH_INDEX_BUFFER_INDEX 13
#define OSD_DRAWINDIRECT_BUFFER_INDEX 14
#define OSD_KERNELLIMIT_BUFFER_INDEX 15

#define FRAME_CONST_BUFFER_INDEX 11
#define INDICES_BUFFER_INDEX 2

using namespace OpenSubdiv::OPENSUBDIV_VERSION;

template <> Far::StencilTable const * Osd::convertToCompatibleStencilTable<OpenSubdiv::Far::StencilTable, OpenSubdiv::Far::StencilTable, OpenSubdiv::Osd::MTLContext>(
                                                                                                                                                                           OpenSubdiv::Far::StencilTable const *table, OpenSubdiv::Osd::MTLContext*  /*context*/) {
    // no need for conversion
    // XXX: We don't want to even copy.
    if (not table) return NULL;
    return new Far::StencilTable(*table);
}

using CPUMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Far::StencilTable,
    Osd::CpuEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using mtlMeshType = Osd::Mesh<
    Osd::CPUMTLVertexBuffer,
    Osd::MTLStencilTable,
    Osd::MTLComputeEvaluator,
    Osd::MTLPatchTable,
    Osd::MTLContext>;

using MTLMeshInterface = Osd::MTLMeshInterface;

struct alignas(16) PerFrameConstants
{
    float ModelViewMatrix[16];
    float ProjectionMatrix[16];
    float ModelViewProjectionMatrix[16];
    float ModelViewInverseMatrix[16];
    float TessLevel;
};

struct alignas(16) Light {
    simd::float4 position;
    simd::float4 ambient;
    simd::float4 diffuse;
    simd::float4 specular;
};

static const char* shaderSource =
#include "mtlViewer.gen.h"
;


using Osd::MTLRingBuffer;

#define FRAME_LAG 3
template<typename DataType>
using PerFrameBuffer = MTLRingBuffer<DataType, FRAME_LAG>;

@implementation OSDRenderer {

    MTLRingBuffer<Light, 1> _lightsBuffer;
    
    PerFrameBuffer<PerFrameConstants> _frameConstantsBuffer;
    PerFrameBuffer<MTLQuadTessellationFactorsHalf> _tessFactorsBuffer;
    PerFrameBuffer<unsigned> _patchIndexBuffers[4];
    PerFrameBuffer<uint8_t> _perPatchDataBuffer;
    PerFrameBuffer<uint8_t> _hsDataBuffer;
    PerFrameBuffer<MTLDrawPatchIndirectArguments> _drawIndirectCommandsBuffer;
    
    unsigned _tessFactorOffsets[4];
    unsigned _perPatchDataOffsets[4];
    unsigned _threadgroupSizes[10];
    
    id<MTLComputePipelineState> _computePipelines[10];
    id<MTLRenderPipelineState> _renderPipelines[10];
    id<MTLRenderPipelineState> _controlLineRenderPipelines[10];
    id<MTLRenderPipelineState> _renderControlEdgesPipeline;
    id<MTLDepthStencilState> _readWriteDepthStencilState;
    id<MTLDepthStencilState> _readOnlyDepthStencilState;

    
    Camera _cameraData;
    Osd::MTLContext _context;
    
    std::unique_ptr<MTLMeshInterface> _mesh;
    std::unique_ptr<MTLControlMeshDisplay> _controlMesh;
    std::unique_ptr<Osd::MTLLegacyGregoryPatchTable> _legacyGregoryPatchTable;
    std::unique_ptr<Shape> _shape;
    
    bool _needsRebuild;
    NSString* _osdShaderSource;
    simd::float3 _meshCenter;
    NSMutableArray<NSString*>* _loadedModels;
    bool _doAdaptive;
}

-(Camera*)camera {
    return &_cameraData;
}

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate {
    self = [super init];
    if(self) {
        self.useSingleCrease = true;
        self.useStageIn = !TARGET_OS_EMBEDDED;
        self.endCapMode = kEndCapBSplineBasis;
        self.useScreenspaceTessellation = true;
        self.usePatchClipCulling = false;
        self.usePatchIndexBuffer = false;
        self.usePatchBackfaceCulling = false;
        self.usePrimitiveBackfaceCulling = false;
        self.tessellationMode = kTessellationModeMetal;
        self.refinementLevel = 2;
        self.tessellationLevel = 8;
        self.shadingMode = kShadingModeMaterial;
        
        _delegate = delegate;
        _context.device = [delegate deviceFor:self];
        _context.commandQueue = [delegate commandQueueFor:self];
        
        
        _osdShaderSource = @(shaderSource);

        _needsRebuild = true;
        
        [self _initializeBuffers];
        [self _initializeCamera];
        [self _initializeLights];
        [self _initializeModels];
    }
    return self;
}

-(void)drawFrame:(id<MTLCommandBuffer>)commandBuffer {    
    if(_needsRebuild) {
        [self _rebuildState];
    }
    
    [self _updateState];
    
    if(_doAdaptive) {
        auto computeEncoder = [commandBuffer computeCommandEncoder];
        [self _computeTessFactors:computeEncoder];
        [computeEncoder endEncoding];
    }
    
    auto renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:[_delegate renderPassDescriptorFor: self]];
    
    if(_usePrimitiveBackfaceCulling) {
        [renderEncoder setCullMode:MTLCullModeBack];
    } else {
        [renderEncoder setCullMode:MTLCullModeNone];
    }
    
    [self _renderMesh:renderEncoder];
    [renderEncoder endEncoding];

    _frameConstantsBuffer.next();
    _tessFactorsBuffer.next();
    _patchIndexBuffers[0].next();
    _patchIndexBuffers[1].next();
    _patchIndexBuffers[2].next();
    _patchIndexBuffers[3].next();
    _lightsBuffer.next();
    _perPatchDataBuffer.next();
    _hsDataBuffer.next();
    _drawIndirectCommandsBuffer.next();
}

-(void)_renderMesh:(id<MTLRenderCommandEncoder>)renderCommandEnocder {
    auto buffer = _mesh->BindVertexBuffer();
    assert(buffer);
    
    auto pav = _mesh->GetPatchTable()->GetPatchArrays();
    auto pib = _mesh->GetPatchTable()->GetPatchIndexBuffer();
    
    [renderCommandEnocder setVertexBuffer:buffer offset:0 atIndex:VERTEX_BUFFER_INDEX];
    [renderCommandEnocder setVertexBuffer: pib offset:0 atIndex:INDICES_BUFFER_INDEX];
    [renderCommandEnocder setVertexBuffer:_frameConstantsBuffer offset:0 atIndex:FRAME_CONST_BUFFER_INDEX];
    
    if(_tessellationMode == kTessellationModeMetal)
    {
        [renderCommandEnocder setVertexBuffer:_hsDataBuffer offset:0 atIndex:OSD_PERPATCHTESSFACTORS_BUFFER_INDEX];
        [renderCommandEnocder setVertexBuffer:_perPatchDataBuffer offset:0 atIndex:OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX];
        [renderCommandEnocder setVertexBuffer:_mesh->GetPatchTable()->GetPatchParamBuffer() offset:0 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
        [renderCommandEnocder setVertexBuffer:_perPatchDataBuffer offset:0 atIndex:OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX];
    }
    
    if(_endCapMode == kEndCapLegacyGregory)
    {
        [renderCommandEnocder setVertexBuffer:_legacyGregoryPatchTable->GetQuadOffsetsBuffer() offset:0 atIndex:OSD_QUADOFFSET_BUFFER_INDEX];
        [renderCommandEnocder setVertexBuffer:_legacyGregoryPatchTable->GetVertexValenceBuffer() offset:0 atIndex:OSD_VALENCE_BUFFER_INDEX];
    }
    
    [renderCommandEnocder setFragmentBuffer:_lightsBuffer offset:0 atIndex:0];
    
    for(int i = 0; i < pav.size(); i++)
    {
        auto& patch = pav[i];
        auto d = patch.GetDescriptor();
        auto patchType = d.GetType();
        auto offset = patchType - Far::PatchDescriptor::REGULAR;
        
        if(_tessellationMode == kTessellationModeMetal)
        {
            [renderCommandEnocder setVertexBufferOffset:patch.primitiveIdBase * sizeof(int) * 3 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
        }
        
        [renderCommandEnocder setVertexBufferOffset:patch.indexBase * sizeof(unsigned) atIndex:INDICES_BUFFER_INDEX];
        
        simd::float4 shade{.0f,0.0f,0.0f,1.0f};
        [renderCommandEnocder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
        [renderCommandEnocder setDepthBias:0 slopeScale:1.0 clamp:0];
        [renderCommandEnocder setTriangleFillMode:MTLTriangleFillModeFill];
        
        [renderCommandEnocder setDepthStencilState:_readWriteDepthStencilState];
        [renderCommandEnocder setRenderPipelineState:_renderPipelines[patchType]];
    
        switch(patchType)
        {
            case Far::PatchDescriptor::GREGORY_BOUNDARY:
            case Far::PatchDescriptor::GREGORY:
            case Far::PatchDescriptor::REGULAR:
                [renderCommandEnocder setVertexBufferOffset:_perPatchDataOffsets[offset] atIndex:OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX];
            case Far::PatchDescriptor::GREGORY_BASIS:
                [renderCommandEnocder setTessellationFactorBuffer:_tessFactorsBuffer offset:_tessFactorOffsets[offset] instanceStride:0];
            break;
            default: break;
        }
        
        
        switch(patchType)
        {
            case Far::PatchDescriptor::GREGORY_BASIS:
                if(_useStageIn)
                {
                    if(_usePatchIndexBuffer)
                    {
                        [renderCommandEnocder drawIndexedPatches:d.GetNumControlVertices() patchStart:0 patchCount:patch.GetNumPatches()
                                                 patchIndexBuffer:_patchIndexBuffers[offset] patchIndexBufferOffset:0
                                          controlPointIndexBuffer:pib controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                                    instanceCount:1 baseInstance:0];
                    }
                    else
                    {
                        [renderCommandEnocder drawIndexedPatches:d.GetNumControlVertices() patchStart:0 patchCount:patch.GetNumPatches()
                                                 patchIndexBuffer:nil patchIndexBufferOffset:0
                                          controlPointIndexBuffer:pib controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                                    instanceCount:1 baseInstance:0];
                    }

                    if(_drawWireframe)
                    {
                        simd::float4 shade = {1, 1,1,1};
                        [renderCommandEnocder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                        [renderCommandEnocder setTriangleFillMode:MTLTriangleFillModeLines];
                        [renderCommandEnocder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];
                        
                        if(_usePatchIndexBuffer)
                        {
                            [renderCommandEnocder drawIndexedPatches:d.GetNumControlVertices() patchStart:0 patchCount:patch.GetNumPatches()
                                                    patchIndexBuffer:_patchIndexBuffers[offset] patchIndexBufferOffset:0
                                             controlPointIndexBuffer:pib controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                                       instanceCount:1 baseInstance:0];
                        }
                        else
                        {
                            [renderCommandEnocder drawIndexedPatches:d.GetNumControlVertices() patchStart:0 patchCount:patch.GetNumPatches()
                                                    patchIndexBuffer:nil patchIndexBufferOffset:0
                                             controlPointIndexBuffer:pib controlPointIndexBufferOffset:patch.indexBase * sizeof(unsigned)
                                                       instanceCount:1 baseInstance:0];
                        }
                    }
                    break;
                }
            case Far::PatchDescriptor::REGULAR:
            case Far::PatchDescriptor::GREGORY:
            case Far::PatchDescriptor::GREGORY_BOUNDARY:
            {
#if !TARGET_OS_EMBEDDED
                if(_usePatchIndexBuffer)
                {
                    [renderCommandEnocder drawPatches:d.GetNumControlVertices()
                        patchIndexBuffer:_patchIndexBuffers[offset] patchIndexBufferOffset:0
                          indirectBuffer:_drawIndirectCommandsBuffer indirectBufferOffset: sizeof(MTLDrawPatchIndirectArguments) * offset];
                }
                else
#endif
                {
                    [renderCommandEnocder drawPatches:d.GetNumControlVertices() patchStart:0 patchCount:patch.GetNumPatches()
                        patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1 baseInstance:0];
                }
                
                if(_drawWireframe)
                {
                    simd::float4 shade = {1, 1,1,1};
                    [renderCommandEnocder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                    [renderCommandEnocder setTriangleFillMode:MTLTriangleFillModeLines];
                    [renderCommandEnocder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];
                    
#if !TARGET_OS_EMBEDDED
                    if(_usePatchIndexBuffer)
                    {
                        [renderCommandEnocder drawPatches:d.GetNumControlVertices()
                                         patchIndexBuffer:_patchIndexBuffers[offset] patchIndexBufferOffset:0
                                           indirectBuffer:_drawIndirectCommandsBuffer indirectBufferOffset: sizeof(MTLDrawPatchIndirectArguments) * offset];
                    }
                    else
#endif
                    {
                        [renderCommandEnocder drawPatches:d.GetNumControlVertices() patchStart:0 patchCount:patch.GetNumPatches()
                                         patchIndexBuffer:nil patchIndexBufferOffset:0 instanceCount:1 baseInstance:0];
                    }
                }
            }
            break;
            
            
            case Far::PatchDescriptor::QUADS:
                [renderCommandEnocder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * 6];
                if(_drawWireframe)
                {
                    simd::float4 shade = {1, 1,1,1};
                    [renderCommandEnocder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                    [renderCommandEnocder setTriangleFillMode:MTLTriangleFillModeLines];
                    [renderCommandEnocder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];
                    [renderCommandEnocder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * 6];
                }
            break;
            case Far::PatchDescriptor::TRIANGLES:
                [renderCommandEnocder setFrontFacingWinding:MTLWindingCounterClockwise];
                [renderCommandEnocder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * d.GetNumControlVertices()];
                if(_drawWireframe)
                {
                    simd::float4 shade = {1, 1,1,1};
                    [renderCommandEnocder setFragmentBytes:&shade length:sizeof(shade) atIndex:2];
                    [renderCommandEnocder setTriangleFillMode:MTLTriangleFillModeLines];
                    [renderCommandEnocder setDepthBias:-5 slopeScale:-1.0 clamp:-100.0];
                    [renderCommandEnocder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:patch.GetNumPatches() * d.GetNumControlVertices()];
                }
            break;
            default:
                assert("Unsupported patch type" && 0); break;
        }
        
        if(_drawControlEdges)
        {
            if(_drawControlEdges && _controlLineRenderPipelines[patchType])
            {
                [renderCommandEnocder setRenderPipelineState:_controlLineRenderPipelines[patchType]];
                
                unsigned primPerPatch = 0;
                switch(patchType)
                {
                    case Far::PatchDescriptor::REGULAR:
                        primPerPatch = 48;
                        break;
                    case Far::PatchDescriptor::GREGORY:
                    case Far::PatchDescriptor::GREGORY_BOUNDARY:
                    case Far::PatchDescriptor::GREGORY_BASIS:
                        primPerPatch = 56;
                        break;
                }
                
                [renderCommandEnocder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:patch.GetNumPatches() * primPerPatch];
            }
        }
    }
    
    if(_drawControlEdges)
    {
        [renderCommandEnocder setDepthStencilState:_readOnlyDepthStencilState];
        _controlMesh->Draw(renderCommandEnocder, _mesh->BindVertexBuffer(), _frameConstantsBuffer->ModelViewProjectionMatrix);
    }
}

-(void)_computeTessFactors:(id<MTLComputeCommandEncoder>)computeCommandEncoder {
    auto& patchArray = _mesh->GetPatchTable()->GetPatchArrays();
    
    [computeCommandEncoder setBuffer:_mesh->BindVertexBuffer() offset:0 atIndex:VERTEX_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_mesh->GetPatchTable()->GetPatchIndexBuffer() offset:0 atIndex:CONTROL_INDICES_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_mesh->GetPatchTable()->GetPatchParamBuffer() offset:0 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_perPatchDataBuffer offset:0 atIndex:OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_hsDataBuffer offset:0 atIndex:OSD_PERPATCHTESSFACTORS_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_tessFactorsBuffer offset:0 atIndex:QUAD_TESSFACTORS_INDEX];
    [computeCommandEncoder setBuffer:_frameConstantsBuffer offset:0 atIndex:FRAME_CONST_BUFFER_INDEX];
    [computeCommandEncoder setBuffer:_perPatchDataBuffer offset:0 atIndex:OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX];
    
    if(_legacyGregoryPatchTable)
    {
        [computeCommandEncoder setBuffer:_legacyGregoryPatchTable->GetQuadOffsetsBuffer() offset:0 atIndex:OSD_QUADOFFSET_BUFFER_INDEX];
        [computeCommandEncoder setBuffer:_legacyGregoryPatchTable->GetVertexValenceBuffer() offset:0 atIndex:OSD_VALENCE_BUFFER_INDEX];
    }
    
    for(auto& patch : patchArray)
    {
        auto usefulControlPoints = patch.GetDescriptor().GetNumControlVertices();
        if(patch.GetDescriptor() == Far::PatchDescriptor::GREGORY_BASIS)
            usefulControlPoints = 4;
        
        auto threadsPerThreadgroup = MTLSizeMake(_threadgroupSizes[patch.desc.GetType()], 1, 1);
        auto threadsPerControlPoint = std::max<int>(1, usefulControlPoints / threadsPerThreadgroup.width);
        
        auto groupPerControlPoint = MTLSizeMake(patch.GetNumPatches() * usefulControlPoints, 1, 1);
        
        groupPerControlPoint.width /= threadsPerControlPoint;
        
        groupPerControlPoint.width = (groupPerControlPoint.width + threadsPerThreadgroup.width - 1) & ~(threadsPerThreadgroup.width - 1);
        groupPerControlPoint.width = groupPerControlPoint.width / threadsPerThreadgroup.width;
        
        
        auto groupPerPatch = MTLSizeMake(patch.GetNumPatches(), 1, 1);
        groupPerPatch.width = (groupPerPatch.width + threadsPerThreadgroup.width - 1) & ~(threadsPerThreadgroup.width - 1);
        groupPerPatch.width = groupPerPatch.width / threadsPerThreadgroup.width;
        
        [computeCommandEncoder setBufferOffset:patch.primitiveIdBase * sizeof(int) * 3 atIndex:OSD_PATCHPARAM_BUFFER_INDEX];
        [computeCommandEncoder setBufferOffset:patch.indexBase * sizeof(unsigned) atIndex:INDICES_BUFFER_INDEX];
        
        
        if(_usePatchIndexBuffer)
        {
            [computeCommandEncoder setBuffer:_patchIndexBuffers[patch.desc.GetType() - Far::PatchDescriptor::REGULAR] offset:0 atIndex:OSD_PATCH_INDEX_BUFFER_INDEX];
            [computeCommandEncoder setBuffer:_drawIndirectCommandsBuffer offset:sizeof(MTLDrawPatchIndirectArguments) * (patch.desc.GetType() - Far::PatchDescriptor::REGULAR) atIndex:OSD_DRAWINDIRECT_BUFFER_INDEX];
        }
        
        [computeCommandEncoder setComputePipelineState:_computePipelines[patch.desc.GetType()]];
        
        unsigned kernelExecutionLimit;
        switch(patch.desc.GetType())
        {
            case Far::PatchDescriptor::REGULAR:
                kernelExecutionLimit = patch.GetNumPatches() * patch.desc.GetNumControlVertices();
                [computeCommandEncoder setBufferOffset:_tessFactorOffsets[0] atIndex:QUAD_TESSFACTORS_INDEX];
                [computeCommandEncoder setBufferOffset:_perPatchDataOffsets[0] atIndex:OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX];
            break;
            case Far::PatchDescriptor::GREGORY:
                kernelExecutionLimit = patch.GetNumPatches() * 4;
                [computeCommandEncoder setBufferOffset:_tessFactorOffsets[1] atIndex:QUAD_TESSFACTORS_INDEX];
                [computeCommandEncoder setBufferOffset:_perPatchDataOffsets[1] atIndex:OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX];
                [computeCommandEncoder setBufferOffset:_legacyGregoryPatchTable->GetQuadOffsetsBase(patch.desc.GetType()) * sizeof(int) atIndex:OSD_QUADOFFSET_BUFFER_INDEX];
            break;
            case Far::PatchDescriptor::GREGORY_BOUNDARY:
                kernelExecutionLimit = patch.GetNumPatches() * 4;
                [computeCommandEncoder setBufferOffset:_tessFactorOffsets[2] atIndex:QUAD_TESSFACTORS_INDEX];
                [computeCommandEncoder setBufferOffset:_perPatchDataOffsets[2] atIndex:OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX];
                [computeCommandEncoder setBufferOffset:_legacyGregoryPatchTable->GetQuadOffsetsBase(patch.desc.GetType()) * sizeof(int) atIndex:OSD_QUADOFFSET_BUFFER_INDEX];
            break;
            case Far::PatchDescriptor::GREGORY_BASIS:
                kernelExecutionLimit = patch.GetNumPatches() * 4;
                [computeCommandEncoder setBufferOffset:_tessFactorOffsets[3] atIndex:QUAD_TESSFACTORS_INDEX];
            break;
            default: assert("Unsupported patch type" && 0); break;
        }
        
        [computeCommandEncoder setBytes:&kernelExecutionLimit length:sizeof(kernelExecutionLimit) atIndex:OSD_KERNELLIMIT_BUFFER_INDEX];
        [computeCommandEncoder dispatchThreadgroups:groupPerControlPoint threadsPerThreadgroup:threadsPerThreadgroup];
    }
}

-(void)_rebuildState {
    [self _rebuildModel];
    [self _rebuildBuffers];
    [self _rebuildPipelines];
    
    _needsRebuild = false;
}

-(void)_rebuildModel {
    
    using namespace OpenSubdiv;
    using namespace Sdc;
    using namespace Osd;
    using namespace Far;
    
    auto shapeDesc = &g_defaultShapes[[_loadedModels indexOfObject:_currentModel]];
    _shape.reset(Shape::parseObj(shapeDesc->data.c_str(), shapeDesc->scheme));
    const auto scheme = shapeDesc->scheme;
    
    // create Far mesh (topology)
    Sdc::SchemeType sdctype = GetSdcType(*_shape);
    Sdc::Options sdcoptions = GetSdcOptions(*_shape);
    
    std::unique_ptr<OpenSubdiv::Far::TopologyRefiner> refiner;
    refiner.reset(
                  Far::TopologyRefinerFactory<Shape>::Create(*_shape, Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions)));
    
    // save coarse topology (used for coarse mesh drawing)
    Far::TopologyLevel const & refBaseLevel = refiner->GetLevel(0);
    
    
    // Adaptive refinement currently supported only for catmull-clark scheme
    _doAdaptive = (_tessellationMode == kTessellationModeMetal && scheme == kCatmark);
    bool doSingleCreasePatch = (_useSingleCrease && scheme == kCatmark);
    
    Osd::MeshBitset bits;
    bits.set(Osd::MeshAdaptive, _doAdaptive);
    bits.set(Osd::MeshUseSingleCreasePatch, doSingleCreasePatch);
    bits.set(Osd::MeshEndCapBSplineBasis, _endCapMode == kEndCapBSplineBasis);
    bits.set(Osd::MeshEndCapGregoryBasis, _endCapMode == kEndCapGregoryBasis);
    bits.set(Osd::MeshEndCapLegacyGregory, _endCapMode == kEndCapLegacyGregory);
    
    int level = _refinementLevel;
    int numVertexElements = 3;
    int numVaryingElements = 0;
    
    if(_tessellationMode == kTessellationModeCPU)
    {
        _mesh.reset(new CPUMeshType(
                                    refiner.release(),
                                    numVertexElements,
                                    numVaryingElements,
                                    level, bits, nullptr, &_context));
    }
    else
    {
        _mesh.reset(new mtlMeshType(
                                    refiner.release(),
                                    numVertexElements,
                                    numVaryingElements,
                                    level, bits, nullptr, &_context));
    }
    
    
    MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
    [_delegate setupRenderPipelineState:desc for:self];
    
    const auto vertexDescriptor = desc.vertexDescriptor;
    vertexDescriptor.layouts[0].stride = sizeof(float) * numVertexElements;
    vertexDescriptor.layouts[0].stepFunction = MTLVertexStepFunctionPerVertex;
    vertexDescriptor.layouts[0].stepRate = 1;
    vertexDescriptor.attributes[0].format = MTLVertexFormatFloat3;
    vertexDescriptor.attributes[0].offset = 0;
    vertexDescriptor.attributes[0].bufferIndex = 0;
    
    _controlMesh.reset(new MTLControlMeshDisplay(_context.device, desc));
    _controlMesh->SetTopology(refBaseLevel);
    _controlMesh->SetEdgesDisplay(true);
    _controlMesh->SetVerticesDisplay(false);
    
    _legacyGregoryPatchTable.reset();
    if(_endCapMode == kEndCapLegacyGregory)
    {
        _legacyGregoryPatchTable.reset(Osd::MTLLegacyGregoryPatchTable::Create(_mesh->GetFarPatchTable(),
                                                                          &_context));
    }
    
    std::vector<float> vertexData;
    vertexData.resize(refBaseLevel.GetNumVertices() * numVertexElements);
    _meshCenter = simd::float3{0,0,0};
    
    if(_shape->normals.size())
    {
        for(int i = 0; i < refBaseLevel.GetNumVertices(); i++)
        {
            vertexData[i * numVertexElements + 0] = _shape->verts[i * 3 + 0];
            vertexData[i * numVertexElements + 1] = _shape->verts[i * 3 + 1];
            vertexData[i * numVertexElements + 2] = _shape->verts[i * 3 + 2];
        }
    }
    else
    {
        for(int vertexIdx = 0; vertexIdx < refBaseLevel.GetNumVertices(); vertexIdx++)
        {
            vertexData[vertexIdx * numVertexElements + 0] = _shape->verts[vertexIdx * 3 + 0];
            vertexData[vertexIdx * numVertexElements + 1] = _shape->verts[vertexIdx * 3 + 1];
            vertexData[vertexIdx * numVertexElements + 2] = _shape->verts[vertexIdx * 3 + 2];
        }
        
        for(auto faceIdx = 0; faceIdx < refBaseLevel.GetNumFaces(); faceIdx++)
        {
            auto faceIndices = refBaseLevel.GetFaceVertices(faceIdx);
            simd::float3 v[4];
            for(int faceVert = 0; faceVert < faceIndices.size(); faceVert++)
            {
                memcpy(v + faceVert, _shape->verts.data() + faceIndices[faceVert] * 3, sizeof(float) * 3);
            }
        }
    }
    
    
    for(auto vertexIdx = 0; vertexIdx < refBaseLevel.GetNumVertices(); vertexIdx++)
    {
        _meshCenter[0] += vertexData[vertexIdx * numVertexElements + 0];
        _meshCenter[1] += vertexData[vertexIdx * numVertexElements + 1];
        _meshCenter[2] += vertexData[vertexIdx * numVertexElements + 2];
    }
    
    
    _meshCenter /= (_shape->verts.size() / 3);
    _mesh->UpdateVertexBuffer(vertexData.data(), 0, refBaseLevel.GetNumVertices());
    _mesh->Refine();
    _mesh->Synchronize();
}

-(void)_updateState {
    [self _updateCamera];
    auto pData = _frameConstantsBuffer.data();
    
    pData->TessLevel = _tessellationLevel;
    
    {
        for(auto& patch : _mesh->GetPatchTable()->GetPatchArrays())
        {
            if(_usePatchIndexBuffer)
            {
                MTLDrawPatchIndirectArguments* drawCommand = _drawIndirectCommandsBuffer.data();
                drawCommand[patch.desc.GetType() - Far::PatchDescriptor::REGULAR].baseInstance = 0;
                drawCommand[patch.desc.GetType() - Far::PatchDescriptor::REGULAR].instanceCount = 1;
                drawCommand[patch.desc.GetType() - Far::PatchDescriptor::REGULAR].patchCount = 0;
                drawCommand[patch.desc.GetType() - Far::PatchDescriptor::REGULAR].patchStart = 0;
            }
        }
        
        if(_usePatchIndexBuffer)
        {
            _drawIndirectCommandsBuffer.markModified();
        }
    }

    _frameConstantsBuffer.markModified();
}

-(void)_rebuildBuffers {
    auto totalPatches = 0;
    auto totalVertices = 0;
    auto totalPatchDataSize = 0;
    
    if(_usePatchIndexBuffer)
    {
        _drawIndirectCommandsBuffer.alloc(_context.device, 4, @"draw patch indirect commands");
    }
    
    if(_tessellationMode == kTessellationModeMetal)
    {
        auto& patchArray = _mesh->GetPatchTable()->GetPatchArrays();
        for(auto& patch : patchArray)
        {
            auto patchDescriptor = patch.GetDescriptor();
            
            switch(patch.desc.GetType())
            {
                case Far::PatchDescriptor::REGULAR: {
                    _patchIndexBuffers[0].alloc(_context.device, patch.GetNumPatches(), @"regular patch indices", MTLResourceStorageModePrivate);
                    _tessFactorOffsets[0] = totalPatches * sizeof(MTLQuadTessellationFactorsHalf);
                    _perPatchDataOffsets[0] = totalPatchDataSize;
                    float elementFloats = 3;
                    if(_useSingleCrease)
                        elementFloats += 6;
                    
                    totalPatchDataSize += elementFloats * sizeof(float) * patch.GetNumPatches() * patch.desc.GetNumControlVertices();
                }
                break;
                case Far::PatchDescriptor::GREGORY:
                _patchIndexBuffers[1].alloc(_context.device, patch.GetNumPatches(), @"gregory patch indices", MTLResourceStorageModePrivate);
                _tessFactorOffsets[1] = totalPatches * sizeof(MTLQuadTessellationFactorsHalf);
                _perPatchDataOffsets[1] = totalPatchDataSize;
                totalPatchDataSize += sizeof(float) * 4 * 8 * patch.GetNumPatches() * patch.desc.GetNumControlVertices();
                break;
                case Far::PatchDescriptor::GREGORY_BOUNDARY:
                _patchIndexBuffers[2].alloc(_context.device, patch.GetNumPatches(), @"gregory boundry patch indices", MTLResourceStorageModePrivate);
                _tessFactorOffsets[2] = totalPatches * sizeof(MTLQuadTessellationFactorsHalf);
                _perPatchDataOffsets[2] = totalPatchDataSize;
                totalPatchDataSize += sizeof(float) * 4 * 8 * patch.GetNumPatches() * patch.desc.GetNumControlVertices();
                break;
                case Far::PatchDescriptor::GREGORY_BASIS:
                _patchIndexBuffers[3].alloc(_context.device, patch.GetNumPatches(), @"gregory basis patch indices", MTLResourceStorageModePrivate);
                _tessFactorOffsets[3] = totalPatches * sizeof(MTLQuadTessellationFactorsHalf);
                _perPatchDataOffsets[3] = totalPatchDataSize;
                //Improved basis doesn't have per-patch-per-vertex data.
                break;
            }
            
            totalPatches += patch.GetNumPatches();
            totalVertices += patch.GetDescriptor().GetNumControlVertices() * patch.GetNumPatches();
        }
        
        _perPatchDataBuffer.alloc(_context.device, totalPatchDataSize, @"per patch data", MTLResourceStorageModePrivate);
        _hsDataBuffer.alloc(_context.device, 20 * sizeof(float) * totalPatches, @"hs constant data", MTLResourceStorageModePrivate);
        _tessFactorsBuffer.alloc(_context.device, totalPatches, @"tessellation factors buffer", MTLResourceStorageModePrivate);
    
    }
}

-(void)_rebuildPipelines {
    for(int i = 0; i < 10; i++) {
        _computePipelines[i] = nil;
        _renderPipelines[i] = nil;
        _renderControlEdgesPipeline = nil;
    }
    
    Osd::MTLPatchShaderSource shaderSource;
    auto& patchArrays = _mesh->GetPatchTable()->GetPatchArrays();
    for(auto& patch : patchArrays)
    {
        auto type = patch.GetDescriptor().GetType();
        auto& threadsPerThreadgroup = _threadgroupSizes[type];
        threadsPerThreadgroup = 32; //Inital guess of 32
        int usefulControlPoints = patch.GetDescriptor().GetNumControlVertices();
        
        auto compileOptions = [[MTLCompileOptions alloc] init];
        compileOptions.fastMathEnabled = YES;
        
        auto preprocessor = [[NSMutableDictionary alloc] init];

        bool allowsSingleCrease = true;
        std::stringstream shaderBuilder;
#define DEFINE(x,y) preprocessor[@(#x)] = @(y)
        switch(type)
        {
            case Far::PatchDescriptor::QUADS:
                DEFINE(OSD_PATCH_QUADS, 1);
            break;
            case Far::PatchDescriptor::TRIANGLES:
                DEFINE(OSD_PATCH_TRIANGLES, 1);
            break;
            case Far::PatchDescriptor::REGULAR:
                DEFINE(OSD_PATCH_REGULAR, 1);
                DEFINE(CONTROL_POINTS_PER_PATCH, 16);
            break;
            case Far::PatchDescriptor::GREGORY:
                DEFINE(OSD_PATCH_GREGORY, 1);
                DEFINE(CONTROL_POINTS_PER_PATCH, 4);
                usefulControlPoints = 4;
                allowsSingleCrease = false;
            break;
            case Far::PatchDescriptor::GREGORY_BASIS:
                DEFINE(OSD_PATCH_GREGORY_BASIS, 1);
                DEFINE(CONTROL_POINTS_PER_PATCH, 4);
                allowsSingleCrease = false;
                usefulControlPoints = 4;
            break;
            case Far::PatchDescriptor::GREGORY_BOUNDARY:
                DEFINE(OSD_PATCH_GREGORY_BOUNDARY, 1);
                DEFINE(CONTROL_POINTS_PER_PATCH, 4);
                allowsSingleCrease = false;
                usefulControlPoints = 4;
            break;
        }
        
#if TARGET_OS_EMBEDDED
        shaderBuilder << "#define OSD_UV_CORRECTION if(t > 0.5){ ti += 0.01f; } else { ti += 0.01f; }\n";
#endif
        
        //Need to define the input vertex struct so that it's available everywhere.
        shaderBuilder << R"(
                            #include <metal_stdlib>
                            struct OsdInputVertexType {
                                metal::packed_float3 position;
                            };
        )";
        
        shaderBuilder << shaderSource.GetCommonShaderSource();
        shaderBuilder << shaderSource.GetHullShaderSource(type);
        shaderBuilder << _osdShaderSource.UTF8String;
        
        const auto str = shaderBuilder.str();
        
        DEFINE(VERTEX_BUFFER_INDEX,VERTEX_BUFFER_INDEX);
        DEFINE(PATCH_INDICES_BUFFER_INDEX,PATCH_INDICES_BUFFER_INDEX);
        DEFINE(CONTROL_INDICES_BUFFER_INDEX,CONTROL_INDICES_BUFFER_INDEX);
        DEFINE(OSD_PATCHPARAM_BUFFER_INDEX,OSD_PATCHPARAM_BUFFER_INDEX);
        DEFINE(OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX,OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX);
        DEFINE(OSD_PERPATCHTESSFACTORS_BUFFER_INDEX,OSD_PERPATCHTESSFACTORS_BUFFER_INDEX);
        DEFINE(OSD_VALENCE_BUFFER_INDEX,OSD_VALENCE_BUFFER_INDEX);
        DEFINE(OSD_QUADOFFSET_BUFFER_INDEX,OSD_QUADOFFSET_BUFFER_INDEX);
        DEFINE(HULL_BUFFER_INDEX,HULL_BUFFER_INDEX);
        DEFINE(FRAME_CONST_BUFFER_INDEX,FRAME_CONST_BUFFER_INDEX);
        DEFINE(INDICES_BUFFER_INDEX,INDICES_BUFFER_INDEX);
        DEFINE(QUAD_TESSFACTORS_INDEX,QUAD_TESSFACTORS_INDEX);
        DEFINE(OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX,OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX);
        DEFINE(OSD_PATCH_INDEX_BUFFER_INDEX,OSD_PATCH_INDEX_BUFFER_INDEX);
        DEFINE(OSD_DRAWINDIRECT_BUFFER_INDEX,OSD_DRAWINDIRECT_BUFFER_INDEX);
        DEFINE(OSD_KERNELLIMIT_BUFFER_INDEX,OSD_KERNELLIMIT_BUFFER_INDEX);
        DEFINE(OSD_PATCH_ENABLE_SINGLE_CREASE, allowsSingleCrease && _useSingleCrease);
        auto partitionMode = _useScreenspaceTessellation ? MTLTessellationPartitionModeFractionalOdd : MTLTessellationPartitionModePow2;
        DEFINE(OSD_FRACTIONAL_EVEN_SPACING, partitionMode == MTLTessellationPartitionModeFractionalEven);
        DEFINE(OSD_FRACTIONAL_ODD_SPACING, partitionMode == MTLTessellationPartitionModeFractionalOdd); 
#if TARGET_OS_EMBEDDED
        DEFINE(OSD_MAX_TESS_LEVEL, 16);
#else
        DEFINE(OSD_MAX_TESS_LEVEL, 64);
#endif
        DEFINE(USE_STAGE_IN, _useStageIn);
        DEFINE(USE_PTVS_FACTORS, !_useScreenspaceTessellation);
        DEFINE(USE_PTVS_SHARPNESS, 1);
        DEFINE(THREADS_PER_THREADGROUP, threadsPerThreadgroup);
        DEFINE(CONTROL_POINTS_PER_THREAD, std::max<int>(1, usefulControlPoints / threadsPerThreadgroup));
        DEFINE(VERTEX_CONTROL_POINTS_PER_PATCH, patch.desc.GetNumControlVertices());
        DEFINE(OSD_MAX_VALENCE, _mesh->GetMaxValence());
        DEFINE(OSD_NUM_ELEMENTS, 3);
        DEFINE(OSD_ENABLE_BACKPATCH_CULL, _usePatchBackfaceCulling);
        DEFINE(SHADING_TYPE, _shadingMode);
        DEFINE(OSD_USE_PATCH_INDEX_BUFFER, _usePatchIndexBuffer);
        DEFINE(OSD_ENABLE_SCREENSPACE_TESSELLATION, _useScreenspaceTessellation);
        DEFINE(OSD_ENABLE_PATCH_CULL, _usePatchClipCulling);
    
        compileOptions.preprocessorMacros = preprocessor;
        
        NSError* err = nil;
        auto librarySource = [NSString stringWithUTF8String:shaderBuilder.str().data()];
        auto library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:&err];
        if(!library && err) {
            NSLog(@"%s", [err localizedDescription].UTF8String);
        }
        assert(library);
        auto vertexFunction = [library newFunctionWithName:@"vertex_main"];
        auto fragmentFunction = [library newFunctionWithName:@"fragment_main"];
        if(vertexFunction && fragmentFunction)
        {
            
            MTLRenderPipelineDescriptor* pipelineDesc = [[MTLRenderPipelineDescriptor alloc] init];
            pipelineDesc.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
            pipelineDesc.tessellationPartitionMode = partitionMode;
            pipelineDesc.tessellationFactorScaleEnabled = false;
            pipelineDesc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
            
            if(type == Far::PatchDescriptor::GREGORY_BASIS && _useStageIn)
                pipelineDesc.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeUInt32;
            
            [_delegate setupRenderPipelineState:pipelineDesc for:self];
            
            {
                pipelineDesc.fragmentFunction = [library newFunctionWithName:@"fragment_solidcolor"];
                pipelineDesc.vertexFunction = [library newFunctionWithName:@"vertex_lines"];
                
                if(pipelineDesc.vertexFunction)
                    _controlLineRenderPipelines[type] = [_context.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&err];
                else
                    _controlLineRenderPipelines[type] = nil;
            }
            
            pipelineDesc.fragmentFunction = fragmentFunction;
            pipelineDesc.vertexFunction = vertexFunction;
            
            if(_useStageIn)
            {
                auto vertexDesc = pipelineDesc.vertexDescriptor;
                [vertexDesc reset];
                
                if(_doAdaptive)
                {
                    vertexDesc.layouts[OSD_PATCHPARAM_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatch;
                    vertexDesc.layouts[OSD_PATCHPARAM_BUFFER_INDEX].stepRate = 1;
                    vertexDesc.layouts[OSD_PATCHPARAM_BUFFER_INDEX].stride = sizeof(int) * 3;
                    
                    
                    vertexDesc.attributes[10].bufferIndex = OSD_PATCHPARAM_BUFFER_INDEX;
                    vertexDesc.attributes[10].format = MTLVertexFormatInt3;
                    vertexDesc.attributes[10].offset = 0;
                }
                
                switch(type)
                {
                    case Far::PatchDescriptor::REGULAR:
                        vertexDesc.layouts[OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
                        vertexDesc.layouts[OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX].stepRate = 1;
                        vertexDesc.layouts[OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX].stride = sizeof(float) * 3;
                        
                        vertexDesc.attributes[0].bufferIndex = OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX;
                        vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
                        vertexDesc.attributes[0].offset = 0;
                        
                        if(_useSingleCrease)
                        {
                            vertexDesc.layouts[OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX].stride += sizeof(float) * 6;
                            
                            vertexDesc.attributes[1].bufferIndex = OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX;
                            vertexDesc.attributes[1].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[1].offset = sizeof(float) * 3;
                            
                            vertexDesc.attributes[2].bufferIndex = OSD_PERPATCHVERTEXBEZIER_BUFFER_INDEX;
                            vertexDesc.attributes[2].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[2].offset = sizeof(float) * 6;
                        }
                        
                        if(_useScreenspaceTessellation)
                        {
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatch;
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stepRate = 1;
                            vertexDesc.layouts[OSD_PERPATCHTESSFACTORS_BUFFER_INDEX].stride = sizeof(float) * 8;
                            
                            vertexDesc.attributes[5].bufferIndex = OSD_PERPATCHTESSFACTORS_BUFFER_INDEX;
                            vertexDesc.attributes[5].format = MTLVertexFormatFloat4;
                            vertexDesc.attributes[5].offset = 0;
                            
                            vertexDesc.attributes[6].bufferIndex = OSD_PERPATCHTESSFACTORS_BUFFER_INDEX;
                            vertexDesc.attributes[6].format = MTLVertexFormatFloat4;
                            vertexDesc.attributes[6].offset = sizeof(float) * 4;
                        }
                    break;
                    case Far::PatchDescriptor::GREGORY_BOUNDARY:
                    case Far::PatchDescriptor::GREGORY:
                        
                        vertexDesc.layouts[OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
                        vertexDesc.layouts[OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX].stepRate = 1;
                        vertexDesc.layouts[OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX].stride = sizeof(float) * 3 * 5;
                        
                        for(int i = 0; i < 5; i++)
                        {
                            vertexDesc.attributes[i].bufferIndex = OSD_PERPATCHVERTEXGREGORY_BUFFER_INDEX;
                            vertexDesc.attributes[i].format = MTLVertexFormatFloat3;
                            vertexDesc.attributes[i].offset = i * sizeof(float) * 3;
                        }
                    break;
                    case Far::PatchDescriptor::GREGORY_BASIS:
                        vertexDesc.layouts[VERTEX_BUFFER_INDEX].stepFunction = MTLVertexStepFunctionPerPatchControlPoint;
                        vertexDesc.layouts[VERTEX_BUFFER_INDEX].stepRate = 1;
                        vertexDesc.layouts[VERTEX_BUFFER_INDEX].stride = sizeof(float) * 3;
                        
                        vertexDesc.attributes[0].bufferIndex = VERTEX_BUFFER_INDEX;
                        vertexDesc.attributes[0].format = MTLVertexFormatFloat3;
                        vertexDesc.attributes[0].offset = 0;
                    break;
                    case Far::PatchDescriptor::QUADS:
                        //Quads cannot use stage in, due to the need for re-indexing.
                        pipelineDesc.vertexDescriptor = nil;
                    case Far::PatchDescriptor::TRIANGLES:
                        [vertexDesc reset];
                        break;
                }
                
            }
            
            _renderPipelines[type] = [_context.device newRenderPipelineStateWithDescriptor:pipelineDesc error:&err];
            if(!_renderPipelines[type] && err)
            {
                NSLog(@"%s", [[err localizedDescription] UTF8String]);
            }
        }
        
        auto computeFunction = [library newFunctionWithName:@"compute_main"];
        if(computeFunction)
        {
            MTLComputePipelineDescriptor* computeDesc = [[MTLComputePipelineDescriptor alloc] init];
#if MTL_TARGET_IPHONE
            computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
#else
            computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = false;
#endif
            computeDesc.computeFunction = computeFunction;
            
            
            NSError* err;
            
            _computePipelines[type] = [_context.device newComputePipelineStateWithDescriptor:computeDesc options:MTLPipelineOptionNone reflection:nil error:&err];
            
            if(err && _computePipelines[type] == nil)
            {
                NSLog(@"%s", [[err description] UTF8String]);
            }
            
            if(_computePipelines[type].threadExecutionWidth != threadsPerThreadgroup)
            {
                DEFINE(THREADS_PER_THREADGROUP, _computePipelines[type].threadExecutionWidth);
                DEFINE(CONTROL_POINTS_PER_THREAD, std::max<int>(1, usefulControlPoints / _computePipelines[type].threadExecutionWidth));
                
                compileOptions.preprocessorMacros = preprocessor;
                
                library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:nil];
                assert(library);
                
                computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = true;
                computeDesc.computeFunction = [library newFunctionWithName:@"compute_main"];
                
                threadsPerThreadgroup = _computePipelines[type].threadExecutionWidth;
                _computePipelines[type] = [_context.device newComputePipelineStateWithDescriptor:computeDesc options:MTLPipelineOptionNone reflection:nil error:&err];
                if(_computePipelines[type].threadExecutionWidth != threadsPerThreadgroup)
                {
                    computeDesc.threadGroupSizeIsMultipleOfThreadExecutionWidth = false;
                    
                    DEFINE(THREADS_PER_THREADGROUP, threadsPerThreadgroup);
                    DEFINE(CONTROL_POINTS_PER_THREAD, std::max<int>(1, usefulControlPoints / threadsPerThreadgroup));
                    DEFINE(NEEDS_BARRIER, 1);
                    
                    compileOptions.preprocessorMacros = preprocessor;
                    
                    library = [_context.device newLibraryWithSource:librarySource options:compileOptions error:nil];
                    assert(library);
                    
                    _computePipelines[type] = [_context.device newComputePipelineStateWithDescriptor:computeDesc options:MTLPipelineOptionNone reflection:nil error:&err];
                }
            }
        }
    }
    
    MTLDepthStencilDescriptor* depthStencilDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthStencilDesc.depthCompareFunction = MTLCompareFunctionLess;
    
    [_delegate setupDepthStencilState:depthStencilDesc for:self];
    
    depthStencilDesc.depthWriteEnabled = YES;
    _readWriteDepthStencilState = [_context.device newDepthStencilStateWithDescriptor:depthStencilDesc];
    
    depthStencilDesc.depthWriteEnabled = NO;
    _readOnlyDepthStencilState = [_context.device newDepthStencilStateWithDescriptor:depthStencilDesc];
}

-(void)_updateCamera {
    auto pData = _frameConstantsBuffer.data();
    
    identity(pData->ModelViewMatrix);
    translate(pData->ModelViewMatrix, 0, 0, -_cameraData.dollyDistance);
    rotate(pData->ModelViewMatrix, _cameraData.rotationY, 1, 0, 0);
    rotate(pData->ModelViewMatrix, _cameraData.rotationX, 0, 1, 0);
    translate(pData->ModelViewMatrix, -_meshCenter[0], -_meshCenter[2], _meshCenter[1]); // z-up model
    rotate(pData->ModelViewMatrix, -90, 1, 0, 0); // z-up model
    inverseMatrix(pData->ModelViewInverseMatrix, pData->ModelViewMatrix);
    
    identity(pData->ProjectionMatrix);
    perspective(pData->ProjectionMatrix, 45.0, _cameraData.aspectRatio, 0.01f, 500.0);
    multMatrix(pData->ModelViewProjectionMatrix, pData->ModelViewMatrix, pData->ProjectionMatrix);
    
}


-(void)_initializeBuffers {
    _frameConstantsBuffer.alloc(_context.device, 1, @"frame constants");
    _tessFactorsBuffer.alloc(_context.device, 1, @"tessellation factors", MTLResourceStorageModePrivate);
    _lightsBuffer.alloc(_context.device, 2, @"lights");
}

-(void)_initializeCamera {
    _cameraData.dollyDistance = 4;
    _cameraData.rotationY = 30;
    _cameraData.rotationX = 0;
    _cameraData.aspectRatio = 1;
}

-(void)_initializeLights {
    _lightsBuffer[0] = {
        simd::normalize(simd::float4{ 0.5,  0.2f, 1.0f, 0.0f }),
        { 0.1f, 0.1f, 0.1f, 1.0f },
        { 0.7f, 0.7f, 0.7f, 1.0f },
        { 0.8f, 0.8f, 0.8f, 1.0f },
    };
    
    _lightsBuffer[1] = {
        simd::normalize(simd::float4{ -0.8f, 0.4f, -1.0f, 0.0f }),
        {  0.0f, 0.0f,  0.0f, 1.0f },
        {  0.5f, 0.5f,  0.5f, 1.0f },
        {  0.8f, 0.8f,  0.8f, 1.0f }
    };
    
    _lightsBuffer.markModified();
}

-(void)_initializeModels {
    initShapes();
    _loadedModels = [[NSMutableArray alloc] initWithCapacity:g_defaultShapes.size()];
    int i = 0;
    for(auto& shape : g_defaultShapes)
    {
        _loadedModels[i++] = [NSString stringWithUTF8String:shape.name.c_str()];
    }
    _currentModel = _loadedModels[0];
}


//Setters for triggering _needsRebuild on property change

-(void)setEndCapMode:(EndCap)endCapMode {
    _needsRebuild |= endCapMode != _endCapMode;
    _endCapMode = endCapMode;
}

-(void)setUseStageIn:(bool)useStageIn {
    _needsRebuild |= useStageIn != _useStageIn;
    _useStageIn = useStageIn;
}

-(void)setShadingMode:(ShadingMode)shadingMode {
    _needsRebuild |= shadingMode != _shadingMode;
    _shadingMode = shadingMode;
}

-(void)setTessellationMode:(TessellationMode)tessellationMode {
    _needsRebuild |= tessellationMode != _tessellationMode;
    _tessellationMode = tessellationMode;
}

-(void)setCurrentModel:(NSString *)currentModel {
    _needsRebuild |= ![currentModel isEqualToString:_currentModel];
    _currentModel = currentModel;
}

-(void)setRefinementLevel:(unsigned int)refinementLevel {
    _needsRebuild |= refinementLevel != _refinementLevel;
    _refinementLevel = refinementLevel;
}

-(void)setUseSingleCrease:(bool)useSingleCrease {
    _needsRebuild |= useSingleCrease != _useSingleCrease;
    _useSingleCrease = useSingleCrease;
}

-(void)setUsePatchClipCulling:(bool)usePatchClipCulling {
    _needsRebuild |= usePatchClipCulling != _usePatchClipCulling;
    _usePatchClipCulling = usePatchClipCulling;
}

-(void)setUsePatchIndexBuffer:(bool)usePatchIndexBuffer {
    _needsRebuild |= usePatchIndexBuffer != _usePatchIndexBuffer;
    _usePatchIndexBuffer = usePatchIndexBuffer;
}

-(void)setUsePatchBackfaceCulling:(bool)usePatchBackfaceCulling {
    _needsRebuild |= usePatchBackfaceCulling != _usePatchBackfaceCulling;
    _usePatchBackfaceCulling = usePatchBackfaceCulling;
}

-(void)setUseScreenspaceTessellation:(bool)useScreenspaceTessellation {
    _needsRebuild |= useScreenspaceTessellation != _useScreenspaceTessellation;
    _useScreenspaceTessellation = useScreenspaceTessellation;
}

@end
