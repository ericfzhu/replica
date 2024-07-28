import fs from 'fs';
import path from 'path';

export function getModelPaths() {
  const modelsDirectory = path.join(process.cwd(), 'models');
  return fs.readdirSync(modelsDirectory).filter(dir => 
    fs.existsSync(path.join(modelsDirectory, dir, 'model.py'))
  );
}