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
	hasCode: boolean;
	lastModified: number;
}

function getLatestModificationTimestamp(folderPath: string): number {
	let latestTimestamp: number = 0;

	const files: string[] = fs.readdirSync(folderPath);

	files.forEach((file: string) => {
		const filePath: string = path.join(folderPath, file);
		const stats: fs.Stats = fs.statSync(filePath);

		if (stats.isFile()) {
			const fileTimestamp: number = stats.mtime.getTime();
			if (fileTimestamp > latestTimestamp) {
				latestTimestamp = fileTimestamp;
			}
		}
	});

	return latestTimestamp;
}

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
		const hasCode = fs.existsSync(modelPath);

		const latestModificationDate = getLatestModificationTimestamp(folderPath);

		return {
			id: folder,
			...metadata,
			hasPaper,
			hasCode,
			lastModified: latestModificationDate,
		};
	});
	return modelsData;
}

export default async function Home() {
	const models = await getModelsData();

	const sortedModels = models.sort((a, b) => {
		if (a.hasCode !== b.hasCode) {
			return a.hasCode ? -1 : 1;
		}
		return b.lastModified - a.lastModified;
	});
	return (
		<main className="flex min-h-screen flex-col items-center gap-12 bg-white p-24 text-black">
			<section className="flex max-w-3xl flex-col items-center gap-2">
				<h1 className="max-w-4xl text-center text-lg lg:text-3xl font-bold uppercase">Replica</h1>
				<span> Implementations of machine learning papers in PyTorch</span>
			</section>

			<section className="flex max-w-3xl flex-col gap-8">
				{sortedModels.map((model: ModelData) => (
					<div key={model.id} className="">
						<h2 className="text-2xl font-semibold">{model.title}</h2>
						<p className="text-gray-600 italic mb-2">{model.authors}</p>
						{/* <span className="text-gray-500 mb-2">Last modified: {new Date(model.lastModified).toLocaleString()}</span> */}
						<div className="flex text-[#4647F1] gap-4 items-center">
							<Link
								href={model.link}
								className="text-sm flex items-center hover:border-[#4647F1] border-transparent border-b-[1px]"
								target="_blank">
								Abstract
								<IconArrowUpRight />
							</Link>
							{model.hasCode ? (
								<Link
									href={model.id}
									className="text-sm flex items-center gap-2 hover:border-[#4647F1] border-transparent border-b-[1px]">
									Code
									<IconArrowRight />
								</Link>
							) : (
								<span className="text-sm rounded-md text-gray-500 cursor-not-allowed flex items-center gap-2">
									Code
									<IconArrowRight />
								</span>
							)}
							<span className="text-sm rounded-md text-gray-500 cursor-not-allowed flex items-center gap-2">
								Model
								<IconArrowUpRight />
							</span>
						</div>
					</div>
				))}
			</section>
		</main>
	);
}
