import { IconArrowLeft, IconArrowUpRight } from '@tabler/icons-react';
import fs from 'fs';
import 'katex/dist/katex.min.css';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import path from 'path';
import Markdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

import CodeDisplay from '@/components/codeDisplay';

interface PageProps {
	params: { modelName: string };
}

interface Metadata {
	title: string;
	authors: string;
	link: string;
}

export default function ModelPage({ params }: PageProps) {
	const modelDir = path.join(process.cwd(), 'public', 'models', params.modelName);
	const modelPath = path.join(modelDir, 'model.py');
	const descriptionPath = path.join(modelDir, 'description.md');
	const metadataPath = path.join(modelDir, 'metadata.json');

	const modelCode = fs.readFileSync(modelPath, 'utf8');
	const description = fs.existsSync(descriptionPath) ? fs.readFileSync(descriptionPath, 'utf8') : null;
	const metadata: Metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));

	return (
		<main className="px-8 mx-auto py-24">
			<div className="relative mb-6 flex w-full flex-col items-center">
				<Link
					href="/"
					className="absolute left-0 top-1/2 -translate-y-1/2 text-[#4647F1] hover:border-[#4647F1] border-transparent border-b-[1px] flex items-center gap-2">
					<IconArrowLeft />
					Back
				</Link>
				<h1 className="text-4xl font-bold uppercase">Replica</h1>
			</div>
			<h2 className="text-2xl font-semibold">{metadata.title}</h2>
			<p className="text-gray-600 italic mb-2">{metadata.authors}</p>
			<div className="flex items-center gap-2">
				<Link
					href={metadata.link}
					className="text-[#4647F1] text-sm flex items-center hover:border-[#4647F1] border-transparent border-b-[1px] w-fit gap-2"
					target="_blank">
					Abstract
					<IconArrowUpRight />
				</Link>
				<span className="text-sm rounded-md text-gray-500 cursor-not-allowed flex items-center gap-2">
					Model
					<IconArrowUpRight />
				</span>
			</div>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
				<CodeDisplay code={modelCode} language="python" fileName={`${params.modelName}_model.py`} />
				{description && (
					<Markdown className="prose prose-zinc max-w-full" remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>
						{description}
					</Markdown>
				)}
			</div>
		</main>
	);
}

export async function generateStaticParams() {
	const modelsDirectory = path.join(process.cwd(), 'public', 'models');
	const modelNames = fs
		.readdirSync(modelsDirectory)
		.filter(
			(dir) => fs.existsSync(path.join(modelsDirectory, dir, 'model.py')) && fs.existsSync(path.join(modelsDirectory, dir, 'metadata.json')),
		);

	return modelNames.map((modelName) => ({
		modelName: modelName,
	}));
}
