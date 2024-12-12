'use client';

import { IconChevronRight } from '@tabler/icons-react';
import { useState } from 'react';

import CodeDisplay from '@/components/codeDisplay';

import { cn } from '@/lib/utils';

interface CodeSectionsProps {
	modelCode: string;
	mainCode: string | null;
	dataloaderCode: string | null;
	modelName: string;
}

export default function CodeSections({ modelCode, mainCode, dataloaderCode, modelName }: CodeSectionsProps) {
	const [openSections, setOpenSections] = useState({
		model: true,
		main: true,
		dataloader: true,
	});

	const toggleSection = (section: keyof typeof openSections) => {
		setOpenSections((prev) => ({
			...prev,
			[section]: !prev[section],
		}));
	};

	return (
		<div className="flex flex-col gap-4">
			<div className="flex flex-col gap-2">
				<button
					onClick={() => toggleSection('model')}
					className="flex items-center gap-2 text-lg font-semibold text-zinc-800 hover:text-zinc-600">
					<IconChevronRight size={20} className={cn(openSections.model ? 'rotate-90' : 'rotate-0', 'transition-transform')} />
					model.py
				</button>

				<div className={cn('grid px-4 transition-all', openSections.model ? 'grid-rows-[1fr] py-4' : 'grid-rows-[0fr]')}>
					<div className="overflow-hidden">
						<CodeDisplay code={modelCode} language="python" fileName={`${modelName}_model.py`} />
					</div>
				</div>
			</div>

			{mainCode && (
				<div className="flex flex-col gap-2">
					<button
						onClick={() => toggleSection('main')}
						className="flex items-center gap-2 text-lg font-semibold text-zinc-800 hover:text-zinc-600">
						<IconChevronRight size={20} className={cn(openSections.main ? 'rotate-90' : 'rotate-0', 'transition-transform')} />
						main.py
					</button>

					<div className={cn('grid px-4 transition-all', openSections.main ? 'grid-rows-[1fr] py-4' : 'grid-rows-[0fr]')}>
						<div className="overflow-hidden">
							<CodeDisplay code={mainCode} language="python" fileName={`${modelName}_main.py`} />
						</div>
					</div>
				</div>
			)}

			{dataloaderCode && (
				<div className="flex flex-col gap-2">
					<button
						onClick={() => toggleSection('dataloader')}
						className="flex items-center gap-2 text-lg font-semibold text-zinc-800 hover:text-zinc-600">
						<IconChevronRight size={20} className={cn(openSections.dataloader ? 'rotate-90' : 'rotate-0', 'transition-transform')} />
						dataloader.py
					</button>

					<div className={cn('grid px-4 transition-all', openSections.dataloader ? 'grid-rows-[1fr] py-4' : 'grid-rows-[0fr]')}>
						<div className="overflow-hidden">
							<CodeDisplay code={dataloaderCode} language="python" fileName={`${modelName}_dataloader.py`} />
						</div>
					</div>
				</div>
			)}
		</div>
	);
}
