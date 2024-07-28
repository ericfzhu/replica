import fs from 'fs';
import path from 'path';

export function getModelFiles(modelName: string) {
  const modelDir = path.join(process.cwd(), 'models', modelName);
  const modelPath = path.join(modelDir, 'model.py');
  const descriptionPath = path.join(modelDir, 'description.md');

  const modelCode = fs.readFileSync(modelPath, 'utf8');
  const description = fs.existsSync(descriptionPath) 
    ? fs.readFileSync(descriptionPath, 'utf8')
    : null;

  return { modelCode, description };
}