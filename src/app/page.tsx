import { IconArrowRight, IconArrowUpRight } from '@tabler/icons-react';
import fs from 'fs';
import Link from 'next/link';
import path from 'path';

export const dynamic = 'force-static';

interface ModelMetadata {
	title: string;
	authors: string;
	link: string;
}

interface ModelData extends ModelMetadata {
	id: string;
	hasModel: boolean;
	hasPaper: boolean;
	lastModified: number;
}

type ModelsData = ModelData[];

async function getModelsData(): Promise<ModelData[]> {
	const modelsDirectory = path.join(process.cwd(), 'public', 'models');
	const items = fs.readdirSync(modelsDirectory, { withFileTypes: true });

	const modelFolders = items.filter((item) => item.isDirectory()).map((item) => item.name);

	const modelsData: ModelData[] = modelFolders.map((folder) => {
		const folderPath = path.join(modelsDirectory, folder);
		const metadataPath = path.join(folderPath, 'metadata.json');
		const fileContents = fs.readFileSync(metadataPath, 'utf8');
		const metadata: ModelMetadata = JSON.parse(fileContents);

		const paperPath = path.join(folderPath, 'paper.pdf');
		const modelPath = path.join(folderPath, 'model.py');

		const hasPaper = fs.existsSync(paperPath);
		const hasModel = fs.existsSync(modelPath);

		const stats = fs.statSync(folderPath);

		return {
			id: folder,
			...metadata,
			hasPaper,
			hasModel,
			lastModified: stats.mtimeMs,
		};
	});
	return modelsData;
}

export default async function Home() {
	const models = await getModelsData();

	const sortedModels = models.sort((a, b) => {
		if (a.hasModel !== b.hasModel) {
			return a.hasModel ? -1 : 1;
		}
		return b.lastModified - a.lastModified;
	});
	return (
		<main className="flex min-h-screen flex-col items-center gap-12 bg-white p-24 text-black">
			<section className="flex max-w-3xl flex-col items-center gap-2">
				<h1 className="max-w-4xl flex-wrap text-center text-lg lg:text-3xl font-bold uppercase">Replica</h1>
				<span> Implementations of machine learning papers in PyTorch</span>
			</section>

			<section className="flex max-w-3xl flex-col gap-8">
				{sortedModels.map((model: ModelData) => (
					<div key={model.id} className="">
						<h2 className="text-2xl font-semibold">{model.title}</h2>
						<p className="text-gray-600 italic mb-2">{model.authors}</p>
						{/* <span className="text-gray-500 mb-2">Last modified: {new Date(model.lastModified).toLocaleString()}</span> */}
						<div className="flex text-[#4647F1] gap-4 items-center">
							<Link href={model.link} className="text-sm flex items-center" target="_blank">
								Abstract
								<IconArrowUpRight />
							</Link>
							{model.hasPaper ? (
								<Link href={`/models/${model.id}/paper.pdf`} className="text-sm flex items-center">
									Paper
									<IconArrowRight />
								</Link>
							) : (
								<span className="text-sm rounded-md text-gray-500 cursor-not-allowed">Paper</span>
							)}
							{/* {model.hasModel ? (
								<Link href={model.id} className="text-sm">
									Model
								</Link>
							) : ( */}
							<span className="text-sm rounded-md text-gray-500 cursor-not-allowed">Model</span>
							{/* )} */}
						</div>
					</div>
				))}
			</section>
		</main>
	);
}

// export async function getStaticProps() {
// 	const modelsDirectory = path.join(process.cwd(), 'models');
// 	const modelFolders = fs.readdirSync(modelsDirectory);

// 	const modelsData: ModelsData = modelFolders.map((folder) => {
// 		const metadataPath = path.join(modelsDirectory, folder, 'metadata.json');
// 		const fileContents = fs.readFileSync(metadataPath, 'utf8');
// 		const metadata: ModelMetadata = JSON.parse(fileContents);

// 		const modelPath = path.join(modelsDirectory, folder, 'model.py');
// 		const paperPath = path.join(modelsDirectory, folder, 'paper.pdf');

// 		return {
// 			id: folder,
// 			...metadata,
// 			hasModel: fs.existsSync(modelPath),
// 			hasPaper: fs.existsSync(paperPath),
// 		};
// 	});

// 	return {
// 		props: {
// 			models: modelsData,
// 		},
// 	};
// }
