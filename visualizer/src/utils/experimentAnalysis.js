const EXPERIMENT_IDS = ['E1', 'E2', 'E3', 'E4'];

function finiteNumber(value) {
  if (value === null || value === undefined || value === '') return null;

  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function cumulativeFinalValue(rows, key) {
  const values = rows.map(row => finiteNumber(row?.[key]));

  if (
    values.length === 0 ||
    values.some(value => value === null) ||
    values.some((value, index) => index > 0 && value < values[index - 1])
  ) {
    return null;
  }

  return values.at(-1);
}

export function calculateExperimentMetrics(rows) {
  if (!Array.isArray(rows) || rows.length === 0) {
    return {
      bestValidationAccuracy: null,
      bestValidationEpochs: [],
      finalValidationAccuracy: null,
      finalTrainingAccuracy: null,
      finalValidationLoss: null,
      totalOptimizerUpdates: null,
      recordedTrainingTimeSeconds: null,
      finalGeneralizationGap: null
    };
  }

  const validationRows = rows
    .map(row => ({
      accuracy: finiteNumber(row?.valAccuracy),
      epoch: finiteNumber(row?.epoch)
    }))
    .filter(row => row.accuracy !== null);

  const bestValidationAccuracy = validationRows.length > 0
    ? Math.max(...validationRows.map(row => row.accuracy))
    : null;

  const bestValidationEpochs = bestValidationAccuracy === null
    ? []
    : validationRows
      .filter(row => row.accuracy === bestValidationAccuracy && row.epoch !== null)
      .map(row => row.epoch);

  const finalRow = rows.at(-1);
  const finalValidationAccuracy = finiteNumber(finalRow?.valAccuracy);
  const finalTrainingAccuracy = finiteNumber(finalRow?.trainAccuracy);
  const finalValidationLoss = finiteNumber(finalRow?.valLoss);

  return {
    bestValidationAccuracy,
    bestValidationEpochs,
    finalValidationAccuracy,
    finalTrainingAccuracy,
    finalValidationLoss,
    totalOptimizerUpdates: cumulativeFinalValue(rows, 'optimizerUpdates'),
    recordedTrainingTimeSeconds: cumulativeFinalValue(rows, 'elapsedSeconds'),
    finalGeneralizationGap:
      finalTrainingAccuracy === null || finalValidationAccuracy === null
        ? null
        : finalTrainingAccuracy - finalValidationAccuracy
  };
}

function findWinners(metricsByExperiment, metric, direction) {
  const values = EXPERIMENT_IDS
    .map(id => ({ id, value: finiteNumber(metricsByExperiment[id]?.[metric]) }))
    .filter(result => result.value !== null);

  if (values.length === 0) return [];

  const bestValue = direction === 'min'
    ? Math.min(...values.map(result => result.value))
    : Math.max(...values.map(result => result.value));

  return values
    .filter(result => result.value === bestValue)
    .map(result => result.id);
}

function differencesFromE1(metricsByExperiment, metric) {
  const baseline = finiteNumber(metricsByExperiment.E1?.[metric]);

  return Object.fromEntries(EXPERIMENT_IDS.map(id => {
    const value = finiteNumber(metricsByExperiment[id]?.[metric]);
    return [id, baseline === null || value === null ? null : value - baseline];
  }));
}

export function analyzeExperiments(experimentData) {
  const metrics = Object.fromEntries(EXPERIMENT_IDS.map(id => [
    id,
    calculateExperimentMetrics(experimentData?.[id])
  ]));

  return {
    metrics,
    winners: {
      bestValidationAccuracy: findWinners(metrics, 'bestValidationAccuracy', 'max'),
      finalValidationAccuracy: findWinners(metrics, 'finalValidationAccuracy', 'max'),
      finalTrainingAccuracy: findWinners(metrics, 'finalTrainingAccuracy', 'max'),
      finalValidationLoss: findWinners(metrics, 'finalValidationLoss', 'min'),
      totalOptimizerUpdates: findWinners(metrics, 'totalOptimizerUpdates', 'min'),
      recordedTrainingTimeSeconds: findWinners(
        metrics,
        'recordedTrainingTimeSeconds',
        'min'
      )
    },
    accuracyDifferences: {
      bestValidationAccuracy: differencesFromE1(metrics, 'bestValidationAccuracy'),
      finalValidationAccuracy: differencesFromE1(metrics, 'finalValidationAccuracy')
    }
  };
}