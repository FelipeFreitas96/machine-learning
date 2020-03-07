const tf = require('@tensorflow/tfjs');

async function regressaoLinear(xArr, yArr) {
    const model = tf.sequential();
    const inputLayer = tf.layers.dense({ units: 1, inputShape: [1] });
    model.add(inputLayer);
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });

    const ylen = yArr.length;
    const xlen = xArr.length;

    const x = tf.tensor(xArr.slice(0, ylen), [ylen, 1]);
    const y = tf.tensor(yArr, [ylen, 1]);
    const input = tf.tensor(xArr.slice(ylen, xlen), [xlen-ylen, 1]);
    
    await model.fit(x, y, { epochs: 800 });
    
    const output = model.predict(input).ceil().flatten();
    return y.flatten().concat(output);
}

/* 
    Resultado:
        Tensor
            [1, 3, 5, 7, 9, 11]
*/

(async () => {
    (await regressaoLinear([1, 2, 3, 4, 5, 6], [1, 3, 5])).print()
})();