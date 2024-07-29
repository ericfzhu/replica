/** @type {import('next').NextConfig} */
const nextConfig = {
	output: 'export',
	reactStrictMode: true,
	swcMinify: true,
	images: {
		unoptimized: true,
	},
	webpack(config) {
		config.resolve.fallback = {
			// if you miss it, all the other options in fallback, specified
			// by next.js will be dropped.
			...config.resolve.fallback,

			fs: false, // the solution
		};

		return config;
	},
};

export default nextConfig;
