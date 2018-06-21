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

MTS_NAMESPACE_BEGIN

class TSVBRDFEvaluator {
public:
	struct Polynom {
		static const int DEGREE = 6;
		Float coefs[DEGREE + 1];
		Float eval(Float t) const {
			Float res = 0.0f;
			for (int i = DEGREE; i >= 0; --i)
				res = res * t + coefs[i];
			return res;
		}
	};

	struct Parameter {
		Polynom phi;
		cv::Mat factors[4];
	};

	void load(const std::string &filepath) {
		
		// Images.
		cv::Mat img;
		const char factorChars[] = { 'A', 'B', 'C', 'D' };

		// Kd.
		for (int f = 0; f < 4; ++f) {
			img = cv::imread(filepath + "/Kd-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
			for (int c = 0; c < 3; ++c)
				cv::extractChannel(img, m_Kd[c].factors[f], c);
		}

		// Ks.
		for (int f = 0; f < 4; ++f)
			m_Ks.factors[f] = cv::imread(filepath + "/Ks-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);

		// Sigma.
		for (int f = 0; f < 4; ++f) {
			m_sigma.factors[f] = cv::imread(filepath + "/Sigma-" + factorChars[f] + ".exr", CV_LOAD_IMAGE_UNCHANGED);
		}

		// Resolution.
		m_width = img.size().width;
		m_height = img.size().height;

		// Phi.
		Float coef;
		std::string line, value;
		std::ifstream file(filepath + "/phi.txt");

		// Kd.
		for (int c = 0; c < 3; ++c) {
			getline(file, line);
			std::istringstream ss(line);
			for (int i = 0; i <= Polynom::DEGREE; ++i) {
				ss >> coef;
				m_Kd[c].phi.coefs[i] = coef;
			}
		}

		// Ks.
		{
			getline(file, line);
			std::istringstream ss(line);
			for (int i = 0; i <= Polynom::DEGREE; ++i) {
				ss >> coef;
				m_Ks.phi.coefs[i] = coef;
			}
		}

		// Sigma.
		{
			getline(file, line);
			std::istringstream ss(line);
			for (int i = 0; i <= Polynom::DEGREE; ++i) {
				ss >> coef;
				m_sigma.phi.coefs[i] = coef;
			}
		}

		file.close();

	}

	inline Float mix(Float x, Float y, Float alpha) const {
		return x * (1 - alpha) + alpha * y;
	}

	Float eval(const Parameter &p, int x, int y, Float t) const {
		std::swap(x, y);
		return Float(p.factors[0].at<float>(y, x) * p.phi.eval((t - p.factors[2].at<float>(y, x))
			/ p.factors[1].at<float>(y, x)) + p.factors[3].at<float>(y, x));
	}

	Float eval(const Parameter &p, Float u, Float v, Float t) const {
		Point2 scaled(fmod(u * (m_width - 1), m_width), fmod(v * (m_height - 1), m_height));
		Point2 upper(ceil(scaled.x), ceil(scaled.y));
		Point2 lower(floor(scaled.x), floor(scaled.y));
		Point2 diff(upper.x - scaled.x, upper.y - scaled.y);
		Float s0 = eval(p, int(lower.x), int(lower.y), t);
		Float s1 = eval(p, int(upper.x), int(lower.y), t);
		Float s2 = eval(p, int(lower.x), int(upper.y), t);
		Float s3 = eval(p, int(upper.x), int(upper.y), t);
		return mix(mix(s0, s1, diff.x), mix(s2, s3, diff.x), diff.y);
		
	}

	Float getKd(Float u, Float v, Float t, int c) const {
		return eval(m_Kd[2 - c], u, v, t);
	}

	Float getKs(Float u, Float v, Float t) const {
		return std::max(0.0f, eval(m_Ks, u, v, t));
	}

	Float getSigma(Float u, Float v, Float t) const {
		return std::max(0.0f, eval(m_sigma, u, v, t));
	}

private:
	int m_width;
	int m_height;
	Parameter m_Kd[3];
	Parameter m_Ks;
	Parameter m_sigma;
};

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

	Spectrum getDiffuseReflectance(const Intersection &its) const {
		Spectrum s;
		Float r = m_evaluator.getKd(its.uv.x, its.uv.y, m_time, 0);
		Float g = m_evaluator.getKd(its.uv.x, its.uv.y, m_time, 1);
		Float b = m_evaluator.getKd(its.uv.x, its.uv.y, m_time, 2);
		s.fromLinearRGB(r, g, b);
		return s;
	}

	inline Vector reflect(const Vector &wi, const Normal &m) const {
		return 2.0f * dot(wi, m) * Vector(m) - wi;
	}

	Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return Spectrum(0.0f);

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		Spectrum result(0.0f);

		if (hasDiffuse) {
			Spectrum kd = getDiffuseReflectance(bRec.its);
			Spectrum diffuse = kd * INV_PI * Frame::cosTheta(bRec.wo);
			result += diffuse;
		}

		if (hasSpecular) {
			Vector H = normalize(bRec.wo + bRec.wi);
			Float ks = m_evaluator.getKs(bRec.its.uv.x, bRec.its.uv.y, m_time);
			Float sigma = m_evaluator.getSigma(bRec.its.uv.x, bRec.its.uv.y, m_time);
			Float spec = ks / (4.0f * Frame::cosTheta(bRec.wi)) * exp(-sigma * Frame::cosTheta2(H));
			Spectrum specular;
			specular.fromLinearRGB(spec, spec, spec);
			result += specular;
		}

		return result;
	}

	Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
		if (Frame::cosTheta(bRec.wi) <= 0 ||
			Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
			return 0.0f;

		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		Vector H = normalize(bRec.wo + bRec.wi);

		Float r = m_evaluator.getKd(bRec.its.uv.x, bRec.its.uv.y, m_time, 0);
		Float g = m_evaluator.getKd(bRec.its.uv.x, bRec.its.uv.y, m_time, 1);
		Float b = m_evaluator.getKd(bRec.its.uv.x, bRec.its.uv.y, m_time, 2);
		Float s = m_evaluator.getKs(bRec.its.uv.x, bRec.its.uv.x, m_time);
		Float sigma = m_evaluator.getSigma(bRec.its.uv.x, bRec.its.uv.x, m_time);

		Float pd = std::max(r, std::max(g, b));
		Float ps = s;
		Float scale = 1.0f / (pd + ps);
		pd *= scale;
		ps *= scale;

		Float diffusePdf = warp::squareToCosineHemispherePdf(bRec.wo);
		Float specularPdf = exp(-sigma * Frame::cosTheta2(H)) * Frame::cosTheta(H) / (4.0f * absDot(bRec.wo, H));
		specularPdf *= sigma / M_PI * (1.0f - exp(-sigma));

		if (hasDiffuse && hasSpecular)
			return ps * specularPdf + pd * diffusePdf;
		else if (hasDiffuse)
			return diffusePdf;
		else if (hasSpecular)
			return specularPdf;
		else
			return 0.0f;
	}

	Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
		Float pdf;
		return TSVBRDF::sample(bRec, pdf, sample);
	}

	Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
		Point2 sample(_sample);
		bool hasSpecular = (bRec.typeMask & EGlossyReflection)
			&& (bRec.component == -1 || bRec.component == 0);
		bool hasDiffuse = (bRec.typeMask & EDiffuseReflection)
			&& (bRec.component == -1 || bRec.component == 1);

		Float r = m_evaluator.getKd(bRec.its.uv.x, bRec.its.uv.y, m_time, 0);
		Float g = m_evaluator.getKd(bRec.its.uv.x, bRec.its.uv.y, m_time, 1);
		Float b = m_evaluator.getKd(bRec.its.uv.x, bRec.its.uv.y, m_time, 2);
		Float s = m_evaluator.getKs(bRec.its.uv.x, bRec.its.uv.x, m_time);
		Float sigma = m_evaluator.getSigma(bRec.its.uv.x, bRec.its.uv.x, m_time);

		Float pd = std::max(r, std::max(g, b));
		Float ps = s;
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
				sample.x = (sample.x - ps)
					/ pd;
				choseSpecular = false;
			}
		}

		if (sigma <= 0.0f)
			choseSpecular = false;

		if (choseSpecular) {
			
			/* Sample normal from Gaussian distribution */
			Float sinPhiM, cosPhiM, cosThetaM, sinThetaM;
			math::sincos((2.0f * M_PI) * sample.y, &sinPhiM, &cosPhiM);
			cosThetaM = std::sqrt(-math::fastlog(exp(-sigma) - sample.x * (exp(-sigma) - 1.0f)) / sigma);
			sinThetaM = std::sqrt(std::max((Float)0, 1 - cosThetaM*cosThetaM));

			Normal m = Vector(
				sinThetaM * cosPhiM,
				sinThetaM * sinPhiM,
				cosThetaM
			);

			/* Perfect specular reflection based on the microfacet normal */
			bRec.wo = reflect(bRec.wi, m);
			bRec.sampledComponent = 0;
			bRec.sampledType = EGlossyReflection;
		}
		else {
			bRec.wo = warp::squareToCosineHemisphere(sample);
			bRec.sampledComponent = 0;
			bRec.sampledType = EDiffuseReflection;
		}
		bRec.eta = 1.0f;

		_pdf = pdf(bRec, ESolidAngle);
		

		if (_pdf == 0)
			return Spectrum(0.0f);
		else
			return eval(bRec, ESolidAngle) / _pdf;
	}

	void serialize(Stream *stream, InstanceManager *manager) const {
		BSDF::serialize(stream, manager);
		stream->writeFloat(m_time);
		stream->writeString(m_filepath);
	}

	Float getRoughness(const Intersection &its, int component) const {
		//return std::numeric_limits<Float>::infinity();
		return 1.0f / sqrt(m_evaluator.getSigma(its.uv.x, its.uv.y, m_time));
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
	TSVBRDFEvaluator m_evaluator;
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
