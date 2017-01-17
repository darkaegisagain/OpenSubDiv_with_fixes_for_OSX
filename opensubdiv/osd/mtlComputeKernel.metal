#line 0 "osd/mtlComputeKernel.metal"

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

#include <metal_stdlib>

#ifndef OPENSUBDIV_MTL_COMPUTE_USE_DERIVATIVES
#define OPENSUBDIV_MTL_COMPUTE_USE_DERIVATIVES 0
#endif

using namespace metal;

struct PatchCoord
{
    int arrayIndex;
    int patchIndex;
    int vertIndex;
    float s;
    float t;
};

struct PatchParam
{
    uint field0;
    uint field1;
    float sharpness;
};

struct KernelUniformArgs
{
	int batchStart;
	int batchEnd;

    int srcOffset;
	int dstOffset;

    int3 duDesc;
    int3 dvDesc;
};

struct Vertex {
    float vertexData[LENGTH];
};

void clear(thread Vertex& v) {
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] = 0;
    }
}

Vertex readVertex(int index, device float* vertexBuffer, KernelUniformArgs args) {
    Vertex v;
    int vertexIndex = args.srcOffset + index * SRC_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] = vertexBuffer[vertexIndex + i];
    }
    return v;
}

void writeVertex(int index, Vertex v, device float* vertexBuffer, KernelUniformArgs args) {
    int vertexIndex = args.dstOffset + index * DST_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        vertexBuffer[vertexIndex + i] = v.vertexData[i];
    }
}

void writeVertexSeparate(int index, Vertex v, device float* dstVertexBuffer, KernelUniformArgs args) {
    int vertexIndex = args.dstOffset + index * DST_STRIDE;
    for (int i = 0; i < LENGTH; ++i) {
        dstVertexBuffer[vertexIndex + i] = v.vertexData[i];
    }
}

void addWithWeight(thread Vertex& v, const Vertex src, float weight) {
    for (int i = 0; i < LENGTH; ++i) {
        v.vertexData[i] += weight * src.vertexData[i];
    }
}

void writeDu(int index, Vertex du, device float* duDerivativeBuffer, KernelUniformArgs args)
{
    int duIndex = args.duDesc.x + index * args.duDesc.z;
    for(int i = 0; i < LENGTH; i++)
    {
        duDerivativeBuffer[duIndex + i] = du.vertexData[i];
    }
}

void writeDv(int index, Vertex dv, device float* dvDerivativeBuffer, KernelUniformArgs args)
{
    int dvIndex = args.dvDesc.x + index * args.dvDesc.z;
    for(int i = 0; i < LENGTH; i++)
    {
        dvDerivativeBuffer[dvIndex + i] = dv.vertexData[i];
    }
}

// ---------------------------------------------------------------------------

kernel void eval_stencils(
    uint thread_position_in_grid [[thread_position_in_grid]],
    const device int* sizes [[buffer(SIZES_BUFFER_INDEX)]],
    const device int* offsets [[buffer(OFFSETS_BUFFER_INDEX)]],
    const device int* indices [[buffer(INDICES_BUFFER_INDEX)]],
    const device float* weights [[buffer(WEIGHTS_BUFFER_INDEX)]],
    device float* srcVertices [[buffer(SRC_VERTEX_BUFFER_INDEX)]],
    device float* dstVertexBuffer [[buffer(DST_VERTEX_BUFFER_INDEX)]],
    const device float* duWeights [[buffer(DU_WEIGHTS_BUFFER_INDEX)]],
    const device float* dvWeights [[buffer(DV_WEIGHTS_BUFFER_INDEX)]],
    device float* duDerivativeBuffer [[buffer(DU_DERIVATIVE_BUFFER_INDEX)]],
    device float* dvDerivativeBuffer [[buffer(DV_DERIVATIVE_BUFFER_INDEX)]],
    const constant KernelUniformArgs& args [[buffer(PARAMETER_BUFFER_INDEX)]]
)
{
    auto current  = thread_position_in_grid + args.batchStart;
    if(current >= args.batchEnd)
        return;

    Vertex dst;
    clear(dst);


    auto offset = offsets[current];
    auto size = sizes[current];

    for(auto stencil = 0; stencil < size; stencil++)
    {
        auto vindex = offset + stencil;
        addWithWeight(dst, readVertex(indices[vindex], srcVertices, args), weights[vindex]);
    }

    writeVertex(current, dst, dstVertexBuffer, args);

#if OPENSUBDIV_MTL_COMPUTE_USE_DERIVATIVES
    Vertex du, dv;
    clear(du);
    clear(dv);


    for(auto i = 0; i < size; i++)
    {
        auto src = readVertex(indices[offset + i], srcVertices, args);
        addWithWeight(du, src, duWeights[offset + i]);
        addWithWeight(dv, src, dvWeights[offset + i]);
    }

    writeDu(current, du, duDerivativeBuffer, args);
    writeDv(current, dv, dvDerivativeBuffer, args);
#endif
}


// ---------------------------------------------------------------------------

// PERFORMANCE: stride could be constant, but not as significant as length

//struct PatchArray {
//    int patchType;
//    int numPatches;
//    int indexBase;        // an offset within the index buffer
//    int primitiveIdBase;  // an offset within the patch param buffer
//};
// # of patcharrays is 1 or 2.

void getBSplineWeights(float t, thread float4& point, thread float4& deriv) {
    // The four uniform cubic B-Spline basis functions evaluated at t:
    constexpr float one6th = 1.0f / 6.0f;

    float t2 = t * t;
    float t3 = t * t2;

    point.x = one6th * (1.0f - 3.0f*(t -      t2) -      t3);
    point.y = one6th * (4.0f           - 6.0f*t2  + 3.0f*t3);
    point.z = one6th * (1.0f + 3.0f*(t +      t2  -      t3));
    point.w = one6th * (                                 t3);

    // Derivatives of the above four basis functions at t:
    deriv.x = -0.5f*t2 +      t - 0.5f;
    deriv.y =  1.5f*t2 - 2.0f*t;
    deriv.z = -1.5f*t2 +      t + 0.5f;
    deriv.w =  0.5f*t2;
}

uint getDepth(uint patchBits) {
    return (patchBits & 0xf);
}

float getParamFraction(uint patchBits) {
    uint nonQuadRoot = (patchBits >> 4) & 0x1;
    uint depth = getDepth(patchBits);
    if (nonQuadRoot == 1) {
        return 1.0f / float( 1 << (depth-1) );
    } else {
        return 1.0f / float( 1 << depth );
    }
}

float2 normalizePatchCoord(uint patchBits, float2 uv) {
    float frac = getParamFraction(patchBits);

    uint iu = (patchBits >> 22) & 0x3ff;
    uint iv = (patchBits >> 12) & 0x3ff;

    // top left corner
    float pu = float(iu*frac);
    float pv = float(iv*frac);

    // normalize u,v coordinates
    return float2((uv.x - pu) / frac, (uv.y - pv) / frac);
}

void adjustBoundaryWeights(uint bits, thread float4& sWeights, thread float4& tWeights) {
    uint boundary = ((bits >> 8) & 0xf);

    if ((boundary & 1) != 0) {
        tWeights[2] -= tWeights[0];
        tWeights[1] += 2*tWeights[0];
        tWeights[0] = 0;
    }
    if ((boundary & 2) != 0) {
        sWeights[1] -= sWeights[3];
        sWeights[2] += 2*sWeights[3];
        sWeights[3] = 0;
    }
    if ((boundary & 4) != 0) {
        tWeights[1] -= tWeights[3];
        tWeights[2] += 2*tWeights[3];
        tWeights[3] = 0;
    }
    if ((boundary & 8) != 0) {
        sWeights[2] -= sWeights[0];
        sWeights[1] += 2*sWeights[0];
        sWeights[0] = 0;
    }
}

// ---------------------------------------------------------------------------

kernel void eval_patches(
                         uint thread_position_in_grid [[thread_position_in_grid]],
                         const constant uint4* patchArrays [[buffer(PATCH_ARRAYS_BUFFER_INDEX)]],
                         device PatchCoord* patchCoords [[buffer(PATCH_COORDS_BUFFER_INDEX)]],
                         device int* patchIndices [[buffer(PATCH_INDICES_BUFFER_INDEX)]],
                         device PatchParam* patchParams [[buffer(PATCH_PARAMS_BUFFER_INDEX)]],
                         device float* srcVertexBuffer [[buffer(SRC_VERTEX_BUFFER_INDEX)]],
                         device float* dstVertexBuffer [[buffer(DST_VERTEX_BUFFER_INDEX)]],
                         device float* duDerivativeBuffer [[buffer(DU_DERIVATIVE_BUFFER_INDEX)]],
                         device float* dvDerivativeBuffer [[buffer(DV_DERIVATIVE_BUFFER_INDEX)]],
                         const constant KernelUniformArgs& args [[buffer(PARAMETER_BUFFER_INDEX)]]
                         )
{
    auto current = thread_position_in_grid;
    auto patchCoord = patchCoords[current];
    auto patchIndex = patchIndices[patchCoord.patchIndex];
    auto patchArray = patchArrays[patchCoord.arrayIndex];
    auto patchType = 6; //Regular, this should definitly be passed in as a #DEFINE
    auto numControlVertices = 16; //Bezier, ^^^
    auto patchBits = patchParams[patchIndex].field1; 
    auto uv = normalizePatchCoord(patchBits, float2(patchCoord.s, patchCoord.t));
    auto dScale = float(1 << getDepth(patchBits));

    float wP[20], wDs[20], wDt[20];
    if(patchType == 6) //Regular
    {
        float4 sWeights, tWeights, dsWeights, dtWeights;
        getBSplineWeights(uv.x, sWeights, dsWeights);
        getBSplineWeights(uv.y, tWeights, dtWeights);

        adjustBoundaryWeights(patchBits, sWeights, tWeights);
        adjustBoundaryWeights(patchBits, dsWeights, dtWeights);

        for(auto k = 0; k < 4; k++)
        {
            for(auto l = 0; l < 4; l++)
            {
                wP[4 * k + l] = sWeights[l] * tWeights[k];
                wDs[4 * k + l] = dsWeights[l] * tWeights[k] * dScale;
                wDt[4 * k + l] = sWeights[l] * dtWeights[k] * dScale;
            }
        }
    }
    else //Greg
    {
        // TODO: GREGORY BASIS
    }

    Vertex dst, du, dv;
    clear(dst);

#if OPENSUBDIV_MTL_COMPUTE_USE_DERIVATIVES
    clear(du);
    clear(dv);
#endif

    auto indexBase = patchArray.z + patchCoord.vertIndex;
    for(auto cv = 0; cv < numControlVertices; cv++)
    {
        auto index = patchIndices[indexBase + cv];
        auto src = readVertex(index, srcVertexBuffer, args);
        addWithWeight(dst, src, wP[cv]);

#if OPENSUBDIV_MTL_COMPUTE_USE_DERIVATIVES
        addWithWeight(du, src, wDs[cv]);
        addWithWeight(dv, src, wDt[cv]);
#endif
    }

    writeVertex(current, dst, dstVertexBuffer, args);

#if OPENSUBDIV_MTL_COMPUTE_USE_DERIVATIVES
    writeDu(current, du, duDerivativeBuffer, args);
    writeDv(current, dv, dvDerivativeBuffer, args);
#endif


}
