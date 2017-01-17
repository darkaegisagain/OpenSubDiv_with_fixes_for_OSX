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


#import "ViewController.h"

@implementation OSDView {
    NSArray<NSTouch*>* _activeTouches;
    bool _mouseDown;
    CGPoint _lastMouse;
    NSTrackingRectTag _tag;
    
}

-(void)mouseDown:(NSEvent *)event {
    if(event.buttonNumber == 0) {
        _mouseDown = true;
        _lastMouse = [NSEvent mouseLocation];
    }
    [super mouseDown:event];
}

-(void)mouseUp:(NSEvent *)event {
    if(event.buttonNumber == 0) {
        _mouseDown = false;
        _lastMouse = [NSEvent mouseLocation];
    }
    [super mouseUp:event];
}

-(void)mouseDragged:(NSEvent *)event {
    if(_mouseDown) {
        CGPoint delta;
        auto mouse = [NSEvent mouseLocation];
        delta.x = mouse.x - _lastMouse.x;
        delta.y = mouse.y - _lastMouse.y;
        _lastMouse = mouse;
        
        _controller.osdRenderer.camera->rotationX += delta.x / 2.0;
        _controller.osdRenderer.camera->rotationY -= delta.y / 2.0;
    }
    [super mouseDragged:event];
}

-(void)scrollWheel:(NSEvent *)event {
    _controller.osdRenderer.camera->dollyDistance += event.deltaY / 100.0;
}

@end



#define FRAME_HISTORY 30
@implementation ViewController {
    id<MTLDevice> _device;
    id<MTLCommandQueue> _commandQueue;
    dispatch_semaphore_t _frameSemaphore;
    OSDRenderer* _osdRenderer;
    
    unsigned _currentFrame;
    double _frameBeginTimestamp[FRAME_HISTORY];
    
    NSMagnificationGestureRecognizer* _magnificationGesture;
}

-(void)viewDidLoad {
    
    _device = MTLCreateSystemDefaultDevice();
    _commandQueue = [_device newCommandQueue];
    
    _osdRenderer = [[OSDRenderer alloc] initWithDelegate:self];
    _osdRenderer.drawControlEdges = false;
    
    _frameSemaphore = dispatch_semaphore_create(3);
    
    self.view.device = _device;
    self.view.delegate = self;
    self.view.depthStencilPixelFormat = MTLPixelFormatDepth32Float;
    self.view.clearColor = MTLClearColorMake(0.4245, 0.4167, 0.4245, 1);
    self.view.drawableSize = CGSizeMake(1920, 1080);
    self.view.controller = self;
    self.view.sampleCount = 4; 
    
    
    _osdRenderer.camera->aspectRatio = self.view.bounds.size.width / self.view.bounds.size.height;
    
    _currentFrame = 0;
    
    [_modelPopup removeAllItems];
    [_modelPopup addItemsWithTitles:_osdRenderer.loadedModels];
    [_modelPopup selectItemAtIndex:0];
    [_modelPopup synchronizeTitleAndSelectedItem];
    
    [_tessellationModePopup removeAllItems];
    [_tessellationModePopup addItemsWithTitles:@[ @"CPU", @"Metal" ]];
    [_tessellationModePopup selectItemWithTitle:@"Metal"];
    [_tessellationModePopup synchronizeTitleAndSelectedItem];
    
    [_refinementLevelPopup removeAllItems];
    [_refinementLevelPopup addItemsWithTitles:@[@"1", @"2", @"3", @"4", @"5", @"6", @"7", @"8"]];
    [_refinementLevelPopup selectItemWithTitle:@"2"];
    [_refinementLevelPopup synchronizeTitleAndSelectedItem];
    
    [_tessellationLevelPopup removeAllItems];
    for(int level = 1; level <= 16; level++) {
        [_tessellationLevelPopup addItemWithTitle:[NSString stringWithFormat:@"%d", level]];
    }
    [_tessellationLevelPopup selectItemWithTitle:@"8"];
    [_tessellationLevelPopup synchronizeTitleAndSelectedItem];
    
    [_endcapModePopup removeAllItems];
    [_endcapModePopup addItemsWithTitles:@[@"None", @"BSpline", @"Legacy Gregory", @"Gregory Basis"]];
    [_endcapModePopup selectItemWithTitle:@"BSpline"];
    [_endcapModePopup synchronizeTitleAndSelectedItem];
    
    [_shadingModePopup removeAllItems];
    [_shadingModePopup addItemsWithTitles:@[@"Material", @"Patch Type", @"Normals", @"Patch Coord"]];
    [_shadingModePopup selectItemWithTitle:@"Material"];
    [_shadingModePopup synchronizeTitleAndSelectedItem];
    
    self.controlCageCheckbox.state = false;
    self.patchIndexCheckbox.state = false;
    self.patchClipCullingCheckbox.state = false;
    self.screenspaceTessellationCheckbox.state = true;
    self.backpatchCullingCheckbox.state = false;
    self.backfaceCullingCheckbox.state = true;
    self.singleCreaseCheckbox.state = true;
    self.wireframeCheckbox.state = false;
    
    [self _applyOptions];
}

-(void)_applyOptions {
    _osdRenderer.useSingleCrease = self.singleCreaseCheckbox.state;
    _osdRenderer.usePatchBackfaceCulling = self.backpatchCullingCheckbox.state;
    _osdRenderer.usePrimitiveBackfaceCulling = self.backfaceCullingCheckbox.state;
    _osdRenderer.useScreenspaceTessellation = self.screenspaceTessellationCheckbox.state;
    _osdRenderer.drawWireframe = self.wireframeCheckbox.state;
    _osdRenderer.drawBezierEdges = self.controlCageCheckbox.state;
    _osdRenderer.drawControlEdges = self.controlCageCheckbox.state;
    _osdRenderer.usePatchIndexBuffer = self.patchIndexCheckbox.state;
    _osdRenderer.usePatchClipCulling = self.patchClipCullingCheckbox.state;
    
    _osdRenderer.tessellationMode = (TessellationMode)self.tessellationModePopup.indexOfSelectedItem;
    _osdRenderer.shadingMode = (ShadingMode)self.shadingModePopup.indexOfSelectedItem;
    _osdRenderer.currentModel = self.modelPopup.titleOfSelectedItem;
    _osdRenderer.refinementLevel = self.refinementLevelPopup.titleOfSelectedItem.intValue;
    _osdRenderer.tessellationLevel = self.tessellationLevelPopup.titleOfSelectedItem.floatValue;
    _osdRenderer.endCapMode = (EndCap)self.endcapModePopup.indexOfSelectedItem;
}

-(void)checkboxChanged:(NSButton *)sender {
    if(sender == self.screenspaceTessellationCheckbox) {
        _osdRenderer.useScreenspaceTessellation = sender.state;
    } else if(sender == self.backpatchCullingCheckbox) {
        _osdRenderer.usePatchBackfaceCulling = sender.state;
    } else if(sender == self.backfaceCullingCheckbox) {
        _osdRenderer.usePrimitiveBackfaceCulling = sender.state;
    } else if(sender == self.patchClipCullingCheckbox) {
        _osdRenderer.usePatchClipCulling = sender.state;
    } else if(sender == self.singleCreaseCheckbox) {
        _osdRenderer.useSingleCrease = sender.state;
    } else if(sender == self.wireframeCheckbox) {
        _osdRenderer.drawWireframe = sender.state;
    } else if(sender == self.patchIndexCheckbox) {
        _osdRenderer.usePatchIndexBuffer = sender.state;
    } else if(sender == self.controlCageCheckbox) {
        _osdRenderer.drawBezierEdges = sender.state;
        _osdRenderer.drawControlEdges = sender.state;
    }
}

-(void)popupChanged:(NSPopUpButton*)sender {
    if(sender == self.modelPopup) {
        _osdRenderer.currentModel = sender.titleOfSelectedItem;
    } else if(sender == self.shadingModePopup) {
        _osdRenderer.shadingMode = (ShadingMode)sender.indexOfSelectedItem;
    } else if(sender == self.tessellationModePopup) {
        _osdRenderer.tessellationMode = (TessellationMode)sender.indexOfSelectedItem;
    } else if(sender == self.tessellationLevelPopup) {
        _osdRenderer.tessellationLevel = sender.titleOfSelectedItem.floatValue;
    } else if(sender == self.refinementLevelPopup) {
        _osdRenderer.refinementLevel = sender.titleOfSelectedItem.intValue;
    } else if(sender == self.endcapModePopup) {
        _osdRenderer.endCapMode = (EndCap)sender.indexOfSelectedItem;
    }
}

- (IBAction)sliderChanged:(NSSlider *)sender {
}

-(void)drawInMTKView:(MTKView *)view {
    dispatch_semaphore_wait(_frameSemaphore, DISPATCH_TIME_FOREVER);
    
    auto commandBuffer = [_commandQueue commandBuffer];
    
    [_osdRenderer drawFrame:commandBuffer];
    
    __weak auto blockSemaphore = _frameSemaphore;
    unsigned frameId = _currentFrame % FRAME_HISTORY;
    auto frameBeginTime = CACurrentMediaTime();
    [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> _Nonnull c) {
        dispatch_semaphore_signal(blockSemaphore);
        _frameBeginTimestamp[frameId] = CACurrentMediaTime() - frameBeginTime;
    }];
    
    [commandBuffer presentDrawable:view.currentDrawable];
    [commandBuffer commit];
    
    double avg = 0;
    for(int i = 0; i < FRAME_HISTORY; i++)
        avg += _frameBeginTimestamp[i];
    avg /= FRAME_HISTORY;
    
    _frameTimeLabel.stringValue = [NSString stringWithFormat:@"Frame: %2.2f ms", avg * 1000.0];
    [_frameTimeLabel sizeToFit];
    
    _currentFrame++;
}

-(void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
    _osdRenderer.camera->aspectRatio = size.width / size.height;
}

-(void)setupDepthStencilState:(MTLDepthStencilDescriptor *)descriptor for:(OSDRenderer *)renderer {
    
}

-(void)setupRenderPipelineState:(MTLRenderPipelineDescriptor *)descriptor for:(OSDRenderer *)renderer {
    descriptor.depthAttachmentPixelFormat = self.view.depthStencilPixelFormat;
    descriptor.colorAttachments[0].pixelFormat = self.view.colorPixelFormat;
    descriptor.sampleCount = self.view.sampleCount;
}

-(id<MTLCommandQueue>)commandQueueFor:(OSDRenderer *)renderer {
    return _commandQueue;
}

-(id<MTLDevice>)deviceFor:(OSDRenderer *)renderer {
    return _device;
}

-(MTLRenderPassDescriptor *)renderPassDescriptorFor:(OSDRenderer *)renderer {
    return self.view.currentRenderPassDescriptor;
}
@end
