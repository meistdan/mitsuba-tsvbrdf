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
    return std::max(0.0f, eval(m_albedo[2 - c], u, v, t));
  }

  Float getNormal(Float u, Float v, Float t, int c) const {
    return std::max(0.0f, eval(m_normal[2 - c], u, v, t));
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
    return std::max(0.0f, eval(m_albedo[2 - c], u, v, t));
  }

  Float getNormal(Float u, Float v, Float t, int c) const {
    return std::max(0.0f, eval(m_normal[2 - c], u, v, t));
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
		m_sampleVisible = true;
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

    Float oneMinusReflectivity = colorSpaceDielectricSpecA * (1.0f - metallic);
    Spectrum diffColor = albedo * oneMinusReflectivity;
    Spectrum specColor = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * albedo;

    Spectrum result(0.0f);
    if (hasDiffuse) {
      result += diffColor * INV_PI * Frame::cosTheta(bRec.wo);
    }

    if (hasSpecular) {
      Vector H = normalize(bRec.wo + bRec.wi);
      MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness, m_sampleVisible);
			const Float F = fresnelConductorExact(dot(bRec.wi, H), 0.0f, 1.0f);
			const Float G = distr.G(bRec.wi, bRec.wo, H);
			const Float D = distr.eval(H);
      result += F * G * D / (4.0f * Frame::cosTheta(bRec.wi)) * specColor;
    }
    
    return result;

  }

#define COSINE_PDF 0
  Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
#if COSINE_PDF
    if (!(bRec.typeMask & EDiffuseReflection) || measure != ESolidAngle
      || Frame::cosTheta(bRec.wi) <= 0
      || Frame::cosTheta(bRec.wo) <= 0)
      return 0.0f;
    return warp::squareToCosineHemispherePdf(bRec.wo);
#else
		if (Frame::cosTheta(bRec.wi) <= 1.0e-3f ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		const Float colorSpaceDielectricSpecRgb = 0.04f;
		const Float colorSpaceDielectricSpecA = 1.0f - 0.04f;

		Float roughness = m_evaluator.getRoughness(bRec.its.uv.x, bRec.its.uv.y, m_time);
		Float metallic = m_evaluator.getMetallic(bRec.its.uv.x, bRec.its.uv.y, m_time);
		Float oneMinusReflectivity = colorSpaceDielectricSpecA * (1.0f - metallic);

		Float r = m_evaluator.getAlbedo(bRec.its.uv.x, bRec.its.uv.y, m_time, 0);
		Float g = m_evaluator.getAlbedo(bRec.its.uv.x, bRec.its.uv.y, m_time, 1);
		Float b = m_evaluator.getAlbedo(bRec.its.uv.x, bRec.its.uv.y, m_time, 2);

		Float dr = oneMinusReflectivity * r;
		Float dg = oneMinusReflectivity * g;
		Float db = oneMinusReflectivity * b;

		Float sr = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * r;
		Float sg = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * g;
		Float sb = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * b;

		Float pd = std::max(dr, std::max(dg, db));
		Float ps = std::max(sr, std::max(sg, sb));
		Float scale = 1.0f / (pd + ps);
		pd *= scale;
		ps *= scale;

		Float diffusePdf = warp::squareToCosineHemispherePdf(bRec.wo);

		MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness, m_sampleVisible);
		Vector H = normalize(bRec.wo + bRec.wi);
		Float specularPdf = 0.0f;
		if (m_sampleVisible)
			specularPdf =  distr.eval(H) * distr.smithG1(bRec.wi, H)
			/ (4.0f * Frame::cosTheta(bRec.wi));
		else
			specularPdf = distr.pdf(bRec.wi, H) / (4 * absDot(bRec.wo, H));

		if (diffusePdf < 1.0e-3f) hasDiffuse = false;
		if (specularPdf < 1.0e-3f) hasSpecular = false;

		if (hasDiffuse && hasSpecular)
			return ps * specularPdf + pd * diffusePdf;
		else if (hasDiffuse)
			return diffusePdf;
		else if (hasSpecular)
			return specularPdf;
		else
			return 0.0f;

#endif
  }

  Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
    Float pdf;
    return TSVBRDF::sample(bRec, pdf, sample);
  }

  Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
#if COSINE_PDF
    if (!(bRec.typeMask & EDiffuseReflection) || Frame::cosTheta(bRec.wi) <= 0)
      return Spectrum(0.0f);
    bRec.wo = warp::squareToCosineHemisphere(_sample);
    bRec.sampledComponent = 0;
    bRec.sampledType = EDiffuseReflection;
    bRec.eta = 1.0f;
    _pdf = warp::squareToCosineHemispherePdf(bRec.wo);
    return eval(bRec, ESolidAngle) / _pdf;
#else
		Point2 sample(_sample);
		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		const Float colorSpaceDielectricSpecRgb = 0.04f;
		const Float colorSpaceDielectricSpecA = 1.0f - 0.04f;

		Float roughness = m_evaluator.getRoughness(bRec.its.uv.x, bRec.its.uv.y, m_time);
		Float metallic = m_evaluator.getMetallic(bRec.its.uv.x, bRec.its.uv.y, m_time);
		Float oneMinusReflectivity = colorSpaceDielectricSpecA * (1.0f - metallic);

		Float r = m_evaluator.getAlbedo(bRec.its.uv.x, bRec.its.uv.y, m_time, 0);
		Float g = m_evaluator.getAlbedo(bRec.its.uv.x, bRec.its.uv.y, m_time, 1);
		Float b = m_evaluator.getAlbedo(bRec.its.uv.x, bRec.its.uv.y, m_time, 2);

		Float dr = oneMinusReflectivity * r;
		Float dg = oneMinusReflectivity * g;
		Float db = oneMinusReflectivity * b;

		Float sr = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * r;
		Float sg = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * g;
		Float sb = (1.0f - metallic) * colorSpaceDielectricSpecRgb + metallic * b;

		Float pd = std::max(dr, std::max(dg, db));
		Float ps = std::max(sr, std::max(sg, sb));
		Float scale = 1.0f / (pd + ps);
		pd *= scale;
		ps *= scale;

		if (!hasSpecular && !hasDiffuse)
			return Spectrum(0.0f);

		bool choseSpecular = hasSpecular;

		if (hasDiffuse && hasSpecular) {
			if (sample.x <= ps) {
				sample.x /= ps;
			}
			else {
				sample.x = (sample.x - ps) / pd;
				choseSpecular = false;
			}
		}

		if (choseSpecular) {

			/* Construct the microfacet distribution matching the
			roughness values at the current surface position. */
			MicrofacetDistribution distr(MicrofacetDistribution::EGGX, roughness, m_sampleVisible);

			/* Sample M, the microfacet normal */
			Normal m = distr.sample(bRec.wi, sample, _pdf);

			if (_pdf == 0)
				return Spectrum(0.0f);

			/* Side check */
			if (Frame::cosTheta(bRec.wo) > 0) {
				/* Perfect specular reflection based on the microfacet normal */
				bRec.wo = reflect(bRec.wi, m);
				bRec.sampledComponent = 1;
				bRec.sampledType = EGlossyReflection;
			}
			else {
				bRec.wo = warp::squareToCosineHemisphere(sample);
				bRec.sampledComponent = 0;
				bRec.sampledType = EDiffuseReflection;
			}
		}
		else {
			bRec.wo = warp::squareToCosineHemisphere(sample);
			bRec.sampledComponent = 0;
			bRec.sampledType = EDiffuseReflection;
		}
		bRec.eta = 1.0f;

		_pdf = pdf(bRec, ESolidAngle);

		if (_pdf == 0.0f)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / _pdf;

#endif
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
	bool m_sampleVisible;
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
