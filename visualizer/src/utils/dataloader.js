
export async function loadExperiment(dataset, epochs, experiment) {
  const path = `/data/${dataset}/${epochs}epochs/${experiment.toLowerCase()}.csv`;

  const response = await fetch(path);

  if (!response.ok) {
    throw new Error(`Failed to load: ${path}`);
  }

  const csvText = await response.text();
  const lines = csvText.trim().split(/\r?\n/);

  const headers = lines[0].split(',');

  const data = lines.slice(1).map(line => {
    const values = line.split(',');
    const row = {};

    headers.forEach((header, index) => {
      row[header.trim()] = values[index];
    });

    return {
      epoch: Number(row.epoch),
      trainAccuracy: Number(row.train_accuracy),
      valAccuracy: Number(row.val_accuracy),
      trainLoss: Number(row.train_loss),
      valLoss: Number(row.val_loss),
      learningRate: Number(row.learning_rate),
      batchSize: Number(row.batch_size),
      optimizerUpdates: Number(row.optimizer_updates),
      elapsedSeconds: Number(row.elapsed_seconds),
      lrChangeReason: row.lr_change_reason || 'keep',
      batchChangeReason: row.batch_controller_reason || ''
    };
  });

  if (data.some(row =>
    !Number.isFinite(row.epoch) ||
    !Number.isFinite(row.trainAccuracy) ||
    !Number.isFinite(row.valAccuracy)
  )) {
    throw new Error(`Invalid training data in ${path}`);
  }

  return data;
}
