#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

typedef enum {
    kTessellationModeCPU = 0,
    kTessellationModeMetal,
} TessellationMode;

typedef enum {
    kDisplacementModeHWBilinear = 0,
    kDisplacementModeBilinear,
    kDisplacementModeBiQuadratic,
    KDisplacementModeNone
} DisplacementMode;

typedef enum {
    kNormalModeHWScreenspace = 0,
    kNormalModeScreenspace,
    kNormalModeBiQuadratic,
    kNormalModeBiQuadraticWG,
    kNormalModeSurface
} NormalMode;

typedef enum {
    kColorModePtexNearest = 0,
    kColorModePtexBilinear,
    kColorModePtexHWBilinear,
    kColorModePtexBiQuadratic,
    kColorModePatchType,
    kColorModePatchCoord,
    kColorModeNormal,
    kColorModeNone
} ColorMode;

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
@property (nonatomic) bool useSeamlessMipmap;


@property (nonatomic) float mipmapBias;
@property (nonatomic) float displacementScale;
@property (nonatomic) bool drawWireframe;

@property (nonatomic) NSString* ptexColorFilename;
@property (nonatomic) NSString* ptexDisplacementFilename;
@property (nonatomic) NSString* ptexOcclusionFilename;
@property (nonatomic) NSString* ptexSpecularFilename;

@property (nonatomic) ColorMode colorMode;
@property (nonatomic) NormalMode normalMode;
@property (nonatomic) DisplacementMode displacementMode;
@property (nonatomic) TessellationMode tessellationMode;

@end
