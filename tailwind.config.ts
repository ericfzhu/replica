import type { Config } from 'tailwindcss';

const config: Config = {
	content: ['./src/pages/**/*.{js,ts,jsx,tsx,mdx}', './src/components/**/*.{js,ts,jsx,tsx,mdx}', './src/app/**/*.{js,ts,jsx,tsx,mdx}'],
	theme: {
		extend: {
			colors: {
				accent: {
					100: '#E3F2FE',
					200: '#C7E3FD',
					300: '#AAD0FB',
					400: '#93BEF7',
					500: '#70A3F2',
					600: '#517ED0',
					700: '#385DAE',
					800: '#23408C',
					900: '#152B74',
				},
			},
		},
	},
	plugins: [],
};
export default config;
