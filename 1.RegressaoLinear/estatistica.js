const tf = require('@tensorflow/tfjs');

function regressaoLinear(xArr, yArr) {
    const x = tf.tensor(xArr.slice(0, yArr.length));
    const y = tf.tensor(yArr);

    const step1 = x.mul(y).sum().sub(x.sum().mul(y.sum()).div(x.size));
    const step2 = x.square().sum().sub(x.sum().square().div(x.size));
    const step3 = step1.div(step2);
    const step4 = y.mean().sub(step3.mul(x.mean()));    
    const diff = Math.abs(yArr.length - xArr.length);

    let array = y.arraySync();
    for (let xPred = 0; xPred < diff; xPred++) {
        array.push(
            step3.mul(xArr[yArr.length + xPred]).add(step4).arraySync(),
        );
    }
    
    return tf.tensor(array);
}

/* 
    Resultado:
        Tensor
            [1, 3, 5, 7, 9, 11]
*/

regressaoLinear([1, 2, 3, 4, 5, 6], [1, 3, 5]).print();