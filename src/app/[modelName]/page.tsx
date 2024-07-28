import { IconArrowLeft, IconArrowUpRight } from '@tabler/icons-react';
import fs from 'fs';
import Link from 'next/link';
import path from 'path';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';

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
		<main className="container mx-auto py-24">
			<div className="flex items-center justify-between mb-6">
				<Link href="/" className="text-[#4647F1] hover:border-[#4647F1] border-transparent border-b-[1px] flex items-center gap-2">
					<IconArrowLeft />
					Back
				</Link>
				<h1 className="text-center text-lg lg:text-3xl font-bold uppercase">Replica</h1>
				<div className="w-10"></div> {/* This empty div balances the layout */}
			</div>

			<h2 className="text-2xl font-bold">{metadata.title}</h2>
			<p className="text-gray-600 italic mb-2">Authors: {metadata.authors}</p>
			<Link
				href={metadata.link}
				className="text-[#4647F1] text-sm flex items-center hover:border-[#4647F1] border-transparent border-b-[1px] w-fit"
				target="_blank">
				Abstract
				<IconArrowUpRight />
			</Link>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
				<div>
					{/* <h3 className="text-xl font-semibold mb-2">model.py</h3> */}
					<SyntaxHighlighter language="python" style={atomDark} codeTagProps={{ className: 'text-sm' }}>
						{modelCode}
					</SyntaxHighlighter>
				</div>
				{description && (
					<div>
						<h3 className="text-xl font-semibold mb-2">Description</h3>
						<ReactMarkdown>{description}</ReactMarkdown>
					</div>
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
