#define EXIT_COLOUR(r,g,b) { gl_FragColor = vec4(r,g,b,1); return; }

uniform sampler2D u_Texture;
uniform sampler2D u_EnvTexture;

uniform sampler2D u_Noise;

uniform vec2 u_Resolution;

uniform float u_NumSamples;

uniform float u_Pitch;
uniform float u_Yaw;
uniform float u_Zoom;
uniform float u_Ez;

uniform vec4 u_Rand;

const float PI = 2.*acos(0.0);

struct HitResult {
	
	bool hit;
	float t;
	vec3 p;
	vec3 n;
	vec3 col;
};

HitResult GetDefaultHitResult() {
	return HitResult(
		false,
		1.0/0.0,
		vec3(0),
		vec3(0),
		vec3(1)
	);
}

vec2 dir_to_latlong(vec3 d)
{
	return vec2(
		0.5*(atan(d.x, d.z)/PI + 1),
		asin(d.y)/PI + 0.5
		);
}

mat3 rotateXMatrix(float tx)
{
	float s = sin(tx);
	float c = cos(tx);
	
	return mat3(1.0 , 0.0 , 0.0 , 
				0.0 , c , s , 
				0.0 , -s , c
			   );
}

mat3 rotateYMatrix(float ty)
{
	float s = sin(ty);
	float c = cos(ty);
	
	return mat3(c , 0.0 , -s ,
				0.0 , 1.0 , 0.0 ,  
				s , 0.0 , c
			   );
}

float cooktorr_brdf(vec3 l, vec3 v, vec3 n, float s)
{

	vec3 h = normalize(l + v);
	float VdotH = max(0, dot(v,h));
	float NdotH = max(0, dot(n,h));
	float NdotL = max(0, dot(n,l));
	float NdotV = max(0, dot(n,v));

	float F0 = 0.5;
	float F = F0 + (1-F0)*pow(1-VdotH, 5);

	float spec_pow = s;
	float D = 0.5*(spec_pow+2)/PI*pow(NdotH,spec_pow);
	float G = NdotL*NdotV/max(NdotL, NdotV);

	return 0.25*F*G*D/(NdotL*NdotV);
	// return max(0, 0.25*F*G*D/(NdotL*NdotV)-1);
}

float diffuse_brdf(vec3 l, vec3 v, vec3 n)
{
	return 1.0/PI;
}

// vec3 ACESFilm(vec3 x)
// {
// 	float a = 2.51;
// 	float b = 0.03;
// 	float c = 2.43;
// 	float d = 0.59;
// 	float e = 0.14;
// 	return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0, 1);
// }

void tracePlane(vec3 o, vec3 d, vec3 n, vec3 p0, vec3 col, inout HitResult hr)
{
	//vec3 n = normalize(vec3(0.0, 1.0, 0.0));
	//vec3 p0 = vec3(0.0);
	
	float t = -dot(o - p0, n) / dot(d, n);
	
	if(t > hr.t && hr.hit) return;
	
	hr.t = t;
	
	vec3 p = o + t*d;
	
	//float hit = step(0.0, sign(t)) * step(length(p-p0), 1.0);// * step(1.0+.5*sin(10.*PI*length(p-p0)+10.*iGlobalTime), 1.0);
	//hr.hit = (sign(t)>0.0) && (length(p-p0)<1.0);
	float s = 1.5;
	hr.hit = (sign(t)>0.0) && all(lessThan(p, vec3(s))) && all(greaterThan(p, vec3(-s)));

	if (hr.hit)
	{
		hr.n = n * sign(-dot(n,d));
		hr.p = p;
		hr.col = col;
	}
	
	//return vec4(p, hit);
}

void traceSphere(vec3 o, vec3 d, vec3 c, float r, vec3 col, inout HitResult hr)
{
	vec3 l = c - o;
	float DdotL = dot(d, l);
	float DdotL2 = DdotL * DdotL;
	float LdotL = dot(l, l);
	
	float D = DdotL2 - LdotL + r*r;
	
	if (D < 0.0) return;
	
	float t1 = DdotL + sqrt(D);
	float t2 = DdotL - sqrt(D);
	
	if (t1 < 0.0 && t2 < 0.0) return;
	
	t1 = t1 > 0.0 ? t1 : 2.0 * t2;
	t2 = t2 > 0.0 ? t2 : 2.0 * t1;
		
	float t = min(t1, t2);
	
	if(t > hr.t && hr.hit) return;
	
	hr.t = t;
	hr.hit = true;
	
	vec3 p = o + t * d;
	hr.n = normalize(p - c);
	hr.p = o + t*d;
	hr.col = col;
}


mat3 kMatrixFromAxis(vec3 k)
{
	return mat3( 0 , -k.z , k.y ,
				 k.z , 0 , -k.x ,
				 -k.y , k.x , 0 
			);
}

vec3 sampleNoise(vec2 uv)
{
	return mod(texture2D(u_Noise, uv+mix(u_Rand.xy, u_Rand.zw, u_Rand.x)).xyz+u_Rand.xyz, vec3(1));
}

vec3 randomRay(vec3 n, vec3 d, vec2 uv)
{
	// vec3 rand = mod(vec3(uv+u_Rand.yx,length(uv+u_Rand.wz))*u_Rand.xyz + sampleNoise(uv+u_Rand.zw).xyz, vec3(1));
	vec3 rand = mod(sampleNoise(uv+u_Rand.xy).xyz+u_Rand.xyz, vec3(1));
	// vec3 rand = u_Rand.xyz;
	// vec3 rand = mod(vec3(u_Rand.yx,length(u_Rand.xz))*u_Rand.xyz + sampleNoise(u_Rand.zw).xyz, vec3(1));
	// float x = mod(rand, sqrt(2*N));
	// float y = rand*sqrt(N/2);
	float theta = 2*PI*rand.x;
	float phi = acos(rand.y-1)-PI/2;
	// float phi = PI*rand.y;
	// float phi = PI*pow(rand.y, 2);
	float cp = cos(phi);
	float sp = sin(phi);
	// float cp = rand.y;
	// float sp = sin(acos(cp));
	float ct = cos(theta);
	float st = sin(theta);
	vec3 v = vec3(cp*st, sp, cp*ct);

	// float angle = acos(dot(vec3(0,1,0), n));
	// vec3 axis = cross(vec3(0,1,0), n);

	// mat3 K = kMatrixFromAxis(axis);
	// mat3 R = mat3(1) + sin(angle)*K + (1-cos(angle))*K*K;
	// vec3 r = R*v;

	// vec3 x = normalize(cross(n, mod(rand,vec3(1))));
	vec3 x = normalize(cross(n, sampleNoise(rand.xy+u_Rand.xy).xyz));
	vec3 z = normalize(cross(x, n));
	mat3 M = mat3(x, n, z);
	vec3 r = M*v;


	return r;
}

void traceScene(vec3 o, vec3 d, inout HitResult hr)
{	
	// sphere radius
	float r = 0.2;
	// sphere centre
	vec3 c = vec3(0.0, 0.0, 0.0);

	vec3 pn = normalize(vec3(0.0, 1.0, 0.0));
	vec3 pp0 = vec3(0.0, -r, 0.);
	tracePlane(o, d, pn, pp0, vec3(1), hr);
	
	traceSphere(o, d, c, r, vec3(1), hr);
	
	traceSphere(o, d, c+vec3(1.0, 0.0, 0.0), r, vec3(0.5,1,0), hr);
	traceSphere(o, d, c+vec3(-1.0, 0.0, 0.0), r, vec3(0,1,1), hr);
	traceSphere(o, d, c+vec3(0.0, 0.0, 1.0), r, vec3(1,0,0.5), hr);
	traceSphere(o, d, c+vec3(0.0, 0.0, -1.0), r, vec3(1,.5,0), hr);
}

vec3 sampleRM(vec3 d)
{
	// vec3 l = normalize(vec3(0.5,0.5,1));
	// float a = step(.99, dot(d,l));
	// return mix(0.5*vec3(0.2,0.5,1), 500*vec3(1,0.8,0.5), 1*smoothstep(.99, 1, dot(d,l)));
	
	// return 
	// 	0.1
	// 	+ 500*vec3(1,0.8,0) * smoothstep(.99, 1, dot(d,l))
	// 	+ 50*vec3(0.5,0,1) * smoothstep(.9, 1, dot(d,normalize(vec3(-1,0.2,-0.5))))
	// 	;

	return texture2D(u_EnvTexture, dir_to_latlong(d)).xyz;
}

float main_brdf(vec3 l, vec3 v, vec3 n)
{
	return mix(diffuse_brdf(l, v, n), cooktorr_brdf(l, v, n, 50), 0.5);
	// return 1.0/PI/max(0.1, dot(n, v));
	// return cooktorr_brdf(l, v, n, 1000)/max(0.1, dot(n, v));
}

void main()
{
	vec2 uv = gl_FragCoord.xy / u_Resolution.xy;

	// vec2 d_xy = 0.99*(mod((1+uv)*u_NumSamples, vec2(1.0)) - 0.5);
	// vec2 d_xy = 0.999*(mod(uv+u_Rand.zw*u_NumSamples+u_Rand.yx+sampleNoise(uv).xy, vec2(1.0)) - 0.5);
	vec2 d_xy = 0.99*(mod(uv+u_Rand.zw+u_Rand.yx, vec2(1.0)) - 0.5);
	vec2 d_uv = d_xy / u_Resolution.xy;
	uv += d_uv;
 
	float aspect = u_Resolution.y / u_Resolution.x;
	vec2 uv2 = 2.0 * (uv - 0.5) * vec2(1.0, aspect);
	
	// ray origin
	vec3 o = vec3(0.0, 0.0, u_Zoom);
	// ray direction
	vec3 d = normalize(vec3(uv2, -u_Ez)).xyz;
	
	mat3 Rx = rotateXMatrix(u_Pitch);
	mat3 Ry = rotateYMatrix(u_Yaw);
	
	d = Ry*Rx*d;
	
	o = vec3(0.0, 0.0, 0.0)+Ry*Rx*o;

	HitResult hr = GetDefaultHitResult();

	traceScene(o, d, hr);

	vec3 planeCol = vec3(1,0,1);

	if (hr.hit)
	{
		vec3 d_r = randomRay(hr.n, sampleNoise(uv+hr.p.xy).xyz, uv);
		vec3 r = d_r;
		// vec3 r = normalize(vec3(1,1,0));
		HitResult hr2 = GetDefaultHitResult();
		traceScene(hr.p+0.001*hr.n, r, hr2);

		float brdf = main_brdf(-d, r, hr.n);
		vec3 s0 = hr.col*max(0, dot(hr.n, r))*brdf;

		if (hr2.hit)
		{
			HitResult hr3 = GetDefaultHitResult();
			vec3 r2 = randomRay(hr2.n, sampleNoise(uv+hr.p.xy).xyz, uv);
			traceScene(hr2.p+0.001*hr2.n, r2, hr3);

			// float brdf2 = 0.1*diffuse_brdf(-r, r2, hr2.n) + cooktorr_brdf(-r, r2, hr2.n);
			float brdf2 = main_brdf(-r, r2, hr2.n);
			vec3 s1 = hr2.col*brdf2*max(0,dot(hr2.n,r2))/(1+hr2.t*hr2.t);

			if (hr3.hit)
			{
				vec3 r3 = randomRay(hr3.n, sampleNoise(uv+hr.p.xy).xyz, uv);
				// float brdf3 = 0.1*diffuse_brdf(-r2, r3, hr3.n) + cooktorr_brdf(-r2, r3, hr3.n);
				float brdf3 = main_brdf(-r2, r3, hr3.n);
				vec3 s2 = hr3.col*brdf3*max(0,dot(hr3.n,r3))/(1+hr3.t*hr3.t);
				planeCol = sampleRM(r3)*s2*s1*s0;
			}
			else
			{
				// planeCol = hr.col * hr2.col * f0*f1 * sampleRM(reflect(r, hr2.n));
				planeCol = sampleRM(r2)*s1*s0;
			}
		}
		else
		{
			planeCol = sampleRM(r)*s0;
			// planeCol = 4*vec3(max(0, dot(hr.n, r))/PI);
		}

		// planeCol = r;
	}
	else 
	{
		planeCol = sampleRM(d);
	}

	vec3 out_col;
	out_col = planeCol;

	// accumulate samples
	if (u_NumSamples > 0)
	{
		vec3 tex = texture2D(u_Texture, uv).xyz; // previous frame
		// if (any(isnan(out_col) || isinf(out_col))) out_col = tex;
		// out_col = ( val < 0.0 || 0.0 < val || val == 0.0 ) ? false : true;
		out_col = any(lessThanEqual(out_col, vec3(0))) || any(greaterThanEqual(out_col, vec3(0))) ? out_col : tex;
		
		out_col = (out_col + u_NumSamples*tex) / (u_NumSamples + 1);
		// out_col = clamp(out_col, 0, 1000000);
	}
	
	gl_FragColor = vec4(out_col,1.0);
}