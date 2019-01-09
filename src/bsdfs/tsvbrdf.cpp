/*
This file is part of Mitsuba, a physically based rendering system.

Copyright (c) 2007-2014 by Wenzel Jakob and others.

Mitsuba is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License Version 3
as published by the Free Software Foundation.

Mitsuba is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>

#include <opencv2/opencv.hpp>

#include "microfacet.h"
#include "ior.h"

MTS_NAMESPACE_BEGIN

class TSVBRDFEvaluator {
public:
  virtual void load(const std::string &filepath) = 0;
  virtual Float getAlbedo(Float u, Float v, Float t, int c) const = 0;
  virtual Float getNormal(Float u, Float v, Float t, int c) const = 0;
  virtual Float getRoughness(Float u, Float v, Float t) const = 0;
  virtual Float getMetallic(Float u, Float v, Float t) const = 0;
protected:
  int m_width;
  int m_height;
};

class PolyEvaluator : public  TSVBRDFEvaluator {
public:
  static const int DEGREE = 5;
  struct Parameter {
    cv::Mat coefs[DEGREE + 1];
  };

  void load(const std::string &filepath) {

    // Images.
    cv::Mat img;

    // Base color.
    for (int d = 0; d <= DEGREE; ++d) {
      img = cv::imread(filepath + "/Albedo-" + std::to_string(d) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      for (int c = 0; c < 3; ++c)
        cv::extractChannel(img, m_albedo[c].coefs[d], c);
    }

    // Normal.
    for (int d = 0; d <= DEGREE; ++d) {
      img = cv::imread(filepath + "/Normal-" + std::to_string(d) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      for (int c = 0; c < 3; ++c)
        cv::extractChannel(img, m_normal[c].coefs[d], c);
    }

    // Roughness.
    for (int d = 0; d <= DEGREE; ++d) {
      img = cv::imread(filepath + "/Rougness-" + std::to_string(d) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      cv::extractChannel(img, m_roughness.coefs[d], 0);
    }

    // Metallic.
    for (int d = 0; d <= DEGREE; ++d) {
      img = cv::imread(filepath + "/Metallic-" + std::to_string(d) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      cv::extractChannel(img, m_metallic.coefs[d], 0);
    }

    // Resolution.
    m_width = img.size().width;
    m_height = img.size().height;

  }

  Float eval(const Parameter &p, int x, int y, Float t) const {
    if (x < 0 || x >= m_width) x = math::modulo(x, m_width);
    if (y < 0 || y >= m_height) y = math::modulo(y, m_height);
    Float res = 0.0f;
    for (int i = DEGREE; i >= 0; --i)
      res = res * t + p.coefs[i].at<float>(y, x);
    return res;
  }

  Float eval(const Parameter &p, Float u, Float v, Float t) const {
    if (EXPECT_NOT_TAKEN(!std::isfinite(u) || !std::isfinite(v)))
      return 0.0f;
    u = u * m_width - 0.5f;
    v = v * m_height - 0.5f;
    int xPos = math::floorToInt(u), yPos = math::floorToInt(v);
    Float dx1 = u - xPos, dx2 = 1.0f - dx1,
      dy1 = v - yPos, dy2 = 1.0f - dy1;
    return eval(p, xPos, yPos, t) * dx2 * dy2
      + eval(p, xPos, yPos + 1, t) * dx2 * dy1
      + eval(p, xPos + 1, yPos, t) * dx1 * dy2
      + eval(p, xPos + 1, yPos + 1, t) * dx1 * dy1;
  }

  Float getAlbedo(Float u, Float v, Float t, int c) const {
    return std::max(0.0f, eval(m_albedo[ c], u, v, t));
  }

  Float getNormal(Float u, Float v, Float t, int c) const {
    return std::max(0.0f, eval(m_normal[c], u, v, t));
  }

  Float getRoughness(Float u, Float v, Float t) const {
    return std::max(0.0f, eval(m_roughness, u, v, t));
  }

  Float getMetallic(Float u, Float v, Float t) const {
    return std::max(0.0f, eval(m_metallic, u, v, t));
  }

private:
  Parameter m_albedo[3];
  Parameter m_normal[3];
  Parameter m_roughness;
  Parameter m_metallic;
};

class FrameEvaluator : public  TSVBRDFEvaluator {
public:
  static const int FRAMES = 51;
  struct Parameter {
    cv::Mat frames[FRAMES];
  };

  void load(const std::string &filepath) {

    // Images.
    cv::Mat img;

    // Base color.
    for (int f = 0; f < FRAMES; ++f) {
      img = cv::imread(filepath + "/Albedo-" + std::to_string(f) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      for (int c = 0; c < 3; ++c)
        cv::extractChannel(img, m_albedo[c].frames[f], c);
    }

    // Normal.
    for (int f = 0; f < FRAMES; ++f) {
      img = cv::imread(filepath + "/Normal-" + std::to_string(f) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      for (int c = 0; c < 3; ++c)
        cv::extractChannel(img, m_normal[c].frames[f], c);
    }

    // Roughness.
		for (int f = 0; f < FRAMES; ++f) {
			img = cv::imread(filepath + "/Roughness-" + std::to_string(f) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
			cv::extractChannel(img, m_roughness.frames[f], 0);
		}

    // Metallic.
    for (int f = 0; f < FRAMES; ++f) {
      img = cv::imread(filepath + "/Metallic-" + std::to_string(f) + ".exr", CV_LOAD_IMAGE_UNCHANGED);
      cv::extractChannel(img, m_metallic.frames[f], 0);
    }

    // Resolution.
    m_width = img.size().width;
    m_height = img.size().height;

  }

  Float eval(const Parameter &p, int x, int y, Float t) const {
    if (x < 0 || x >= m_width) x = math::modulo(x, m_width);
    if (y < 0 || y >= m_height) y = math::modulo(y, m_height);
    t = math::clamp(t, 0.0f, 1.0f);
    int i = math::roundToInt(t * (FRAMES - 1));
    return p.frames[i].at<float>(y, x);
  }

  Float eval(const Parameter &p, Float u, Float v, Float t) const {
    if (EXPECT_NOT_TAKEN(!std::isfinite(u) || !std::isfinite(v)))
      return 0.0f;
    u = u * m_width - 0.5f;
    v = v * m_height - 0.5f;
    int xPos = math::floorToInt(u), yPos = math::floorToInt(v);
    Float dx1 = u - xPos, dx2 = 1.0f - dx1,
      dy1 = v - yPos, dy2 = 1.0f - dy1;
    return eval(p, xPos, yPos, t) * dx2 * dy2
      + eval(p, xPos, yPos + 1, t) * dx2 * dy1
      + eval(p, xPos + 1, yPos, t) * dx1 * dy2
      + eval(p, xPos + 1, yPos + 1, t) * dx1 * dy1;
  }

  Float getAlbedo(Float u, Float v, Float t, int c) const {
    return std::max(0.0f, eval(m_albedo[c], u, v, t));
  }

  Float getNormal(Float u, Float v, Float t, int c) const {
    return std::max(0.0f, eval(m_normal[c], u, v, t));
  }

  Float getRoughness(Float u, Float v, Float t) const {
    return std::max(0.0f, eval(m_roughness, u, v, t));
  }

  Float getMetallic(Float u, Float v, Float t) const {
    return std::max(0.0f, eval(m_metallic, u, v, t));
  }

private:
  Parameter m_albedo[3];
  Parameter m_normal[3];
  Parameter m_roughness;
  Parameter m_metallic;
};

//typedef PolyEvaluator Evaluator;
typedef FrameEvaluator Evaluator;

class TSVBRDF : public BSDF {
public:
  TSVBRDF(const Properties &props)
    : BSDF(props) {
    m_time = props.getFloat("time", 0.0f);
    m_filepath = props.getString("filepath", "");
    if (m_filepath.empty())
      Log(EError, "Filepath to TSVBRDF was not specified!");
    m_evaluator.load(m_filepath);
  }

  TSVBRDF(Stream *stream, InstanceManager *manager)
    : BSDF(stream, manager) {
    m_time = stream->readFloat();
    m_filepath = stream->readString();
    m_evaluator.load(m_filepath);
    configure();
  }

  void configure() {
    m_components.clear();
    m_components.push_back(EGlossyReflection | EFrontSide | ESpatiallyVarying);
    m_usesRayDifferentials = true;
    BSDF::configure();
  }

  Spectrum getAlbedo(const Intersection &its) const {
    Spectrum s;
    Float r = m_evaluator.getAlbedo(its.uv.x, its.uv.y, m_time, 0);
    Float g = m_evaluator.getAlbedo(its.uv.x, its.uv.y, m_time, 1);
    Float b = m_evaluator.getAlbedo(its.uv.x, its.uv.y, m_time, 2);
    s.fromLinearRGB(r, g, b);
    return s;
  }

  Normal getNormal(const Intersection &its) const {
    Float x = m_evaluator.getNormal(its.uv.x, its.uv.y, m_time, 0);
    Float y = m_evaluator.getNormal(its.uv.x, its.uv.y, m_time, 1);
    Float z = m_evaluator.getNormal(its.uv.x, its.uv.y, m_time, 2);
    return normalize(Normal(x, y, z));
  }

  inline Vector safeNormalize(const Vector &inVec) const {
    float dp3 = std::max(0.001f, dot(inVec, inVec));
    return inVec * pow(dp3, -0.5f);
  }

  inline Float smoothnessToPerceptualRoughness(Float smoothness) const {
    return (1.0f - smoothness);
  }

  inline Float saturate(Float x) const {
    return std::max(0.0f, std::min(1.0f, x));
  }

  inline Spectrum fresnelTerm(Spectrum F0, Float cosA) const {
    Float t = pow(1.0f - cosA, 5.0f);   // ala Schlick interpoliation
    return F0 + (Spectrum(1.0f)- F0) * t;
  }

  inline Spectrum fresnelLerp(Spectrum F0, Spectrum F90, Float cosA) const {
    Float t = pow(1 - cosA, 5.0f);   // ala Schlick interpoliation
    return (1.0f - t) * F0 + t * F90;
  }

  Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
    if (Frame::cosTheta(bRec.wi) <= 1.0e-3f ||
      Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
      return Spectrum(0.0f);

    bool hasSpecular = (bRec.typeMask & EGlossyReflection)
      && (bRec.component == -1 || bRec.component == 0);
    bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
      && (bRec.component == -1 || bRec.component == 1);

    const Spectrum colorSpaceDielectricSpecRgb(0.04f);
    const Float colorSpaceDielectricSpecA = 1.0f - 0.04f;

    Float roughness = m_evaluator.getRoughness(bRec.its.uv.x, bRec.its.uv.y, m_time);
    Float metallic = m_evaluator.getMetallic(bRec.its.uv.x, bRec.its.uv.y, m_time);
    Spectrum albedo = getAlbedo(bRec.its);
    Normal normal = getNormal(bRec.its);

    Float oneMinusReflectivity = colorSpaceDielectricSpecA * (1.0f - metallic);
    Spectrum diffColor = albedo * oneMinusReflectivity;
    Spectrum specColor = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * albedo;

    Spectrum result(0.0f);
    if (hasDiffuse) {
      result += diffColor * INV_PI * Frame::cosTheta(bRec.wo);
      //result += diffColor * INV_PI * dot(normal, bRec.wo);
    }

    if (hasSpecular) {
      Vector H = normalize(bRec.wo + bRec.wi);
      MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness);
      const Float D = distr.eval(H);
      const Float G = distr.G(bRec.wi, bRec.wo, H);
      const Spectrum F = fresnelConductorExact(dot(bRec.wi, H), 0.0f, 1.0f) * specColor;
      result += F * G * D / (4.0f * Frame::cosTheta(bRec.wi));
      //result += F * G * D / (4.0f * dot(normal, bRec.wi));
    }
    
    return result;

  }

  Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
    if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
      || Frame::cosTheta(bRec.wi) <= 0
      || Frame::cosTheta(bRec.wo) <= 0)
      return 0.0f;
    return warp::squareToCosineHemispherePdf(bRec.wo);
  }

  Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
    Float pdf;
    return TSVBRDF::sample(bRec, pdf, sample);
  }

  Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
    if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
      return Spectrum(0.0f);
    bRec.wo = warp::squareToCosineHemisphere(sample);
    bRec.sampledComponent = 0;
    bRec.sampledType = EDiffuseReflection;
    bRec.eta = 1.0f;
    pdf = warp::squareToCosineHemispherePdf(bRec.wo);
    return eval(bRec, ESolidAngle) / pdf;
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    BSDF::serialize(stream, manager);
    stream->writeFloat(m_time);
    stream->writeString(m_filepath);
  }

  Float getRoughness(const Intersection &its, int component) const {
    //return std::numeric_limits<Float>::infinity();
    return 1.0f / sqrt(m_evaluator.getRoughness(its.uv.x, its.uv.y, m_time));
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "TSVBRDF[" << endl
      << "  filepath = " << m_filepath << "," << endl
      << "  time = " << m_time << endl
      << "]";
    return oss.str();
  }

  Shader *createShader(Renderer *renderer) const;

  MTS_DECLARE_CLASS()
private:
  std::string m_filepath;
  Float m_time;
  Evaluator m_evaluator;
};

// ================ Hardware shader implementation ================

// Fake shader
class TSVBRDFShader : public Shader {
public:
  TSVBRDFShader(Renderer *renderer)
    : Shader(renderer, EBSDFShader) {
    m_reflectance = new ConstantSpectrumTexture(Spectrum(.5f));
    m_reflectanceShader = renderer->registerShaderForResource(m_reflectance.get());
  }

  bool isComplete() const {
    return m_reflectanceShader.get() != NULL;
  }

  void cleanup(Renderer *renderer) {
    renderer->unregisterShaderForResource(m_reflectance.get());
  }

  void putDependencies(std::vector<Shader *> &deps) {
    deps.push_back(m_reflectanceShader.get());
  }

  void generateCode(std::ostringstream &oss,
    const std::string &evalName,
    const std::vector<std::string> &depNames) const {
    oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
      << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
      << "    	return vec3(0.0);" << endl
      << "    return " << depNames[0] << "(uv) * inv_pi * cosTheta(wo);" << endl
      << "}" << endl
      << endl
      << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
      << "    return " << evalName << "(uv, wi, wo);" << endl
      << "}" << endl;
  }

  MTS_DECLARE_CLASS()
private:
  ref<const Texture> m_reflectance;
  ref<Shader> m_reflectanceShader;
};

Shader *TSVBRDF::createShader(Renderer *renderer) const {
  return new TSVBRDFShader(renderer);
}

MTS_IMPLEMENT_CLASS(TSVBRDFShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(TSVBRDF, false, BSDF)
MTS_EXPORT_PLUGIN(TSVBRDF, "Time-spatially varying BRDF")
MTS_NAMESPACE_END
