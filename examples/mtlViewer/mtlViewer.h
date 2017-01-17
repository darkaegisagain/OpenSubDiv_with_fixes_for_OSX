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

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef enum {
    kEndCapNone = 0,
    kEndCapBSplineBasis,
    kEndCapLegacyGregory,
    kEndCapGregoryBasis,
} EndCap;

typedef enum {
    kTessellationModeCPU = 0,
    kTessellationModeMetal,
} TessellationMode;

typedef enum {
    kShadingModeMaterial = 0,
    kShadingModePatchType,
    kShadingModeNormals,
    kShadingModePatchCoord,
} ShadingMode;

typedef struct {
    float rotationX;
    float rotationY;
    float dollyDistance;
    float aspectRatio;
} Camera;

@class OSDRenderer;

@protocol OSDRendererDelegate <NSObject>
-(id<MTLDevice>)deviceFor:(OSDRenderer*)renderer;
-(id<MTLCommandQueue>)commandQueueFor:(OSDRenderer*)renderer;
-(MTLRenderPassDescriptor*)renderPassDescriptorFor:(OSDRenderer*)renderer;
-(void)setupDepthStencilState:(MTLDepthStencilDescriptor*)descriptor for:(OSDRenderer*)renderer;
-(void)setupRenderPipelineState:(MTLRenderPipelineDescriptor*)descriptor for:(OSDRenderer*)renderer;
@end

@interface OSDRenderer : NSObject

-(instancetype)initWithDelegate:(id<OSDRendererDelegate>)delegate;

-(void)drawFrame:(id<MTLCommandBuffer>)commandBuffer;

@property (readonly, nonatomic) id<OSDRendererDelegate> delegate;

@property (nonatomic) unsigned refinementLevel;
@property (nonatomic) float tessellationLevel;

@property (readonly, nonatomic) NSArray<NSString*>* loadedModels;
@property (nonatomic) NSString* currentModel;

@property (readonly, nonatomic) Camera* camera;

@property (nonatomic) bool useScreenspaceTessellation;
@property (nonatomic) bool usePatchIndexBuffer;
@property (nonatomic) bool usePatchBackfaceCulling;
@property (nonatomic) bool usePatchClipCulling;
@property (nonatomic) bool useSingleCrease;
@property (nonatomic) bool useStageIn;
@property (nonatomic) bool usePrimitiveBackfaceCulling;

@property (nonatomic) bool drawControlEdges;
@property (nonatomic) bool drawWireframe;
@property (nonatomic) bool drawBezierEdges;

@property (nonatomic) ShadingMode shadingMode;
@property (nonatomic) EndCap endCapMode;
@property (nonatomic) TessellationMode tessellationMode;

@end
