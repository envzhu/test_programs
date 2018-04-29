/* 
 * Find coefficients of linear function
 */

const step_num = 5000;

window.addEventListener("DOMContentLoaded", function() {
  document.getElementById("prog1").addEventListener("click",function(){
    
    console.log("==========  PROG1 START ==========");
    //y = 0.1 * x + 0.3
    const originA = tf.scalar(0.1);
    const originB = tf.scalar(0.3);
    
    let randomData = [];
    for(let i=0;i<=100;i++){
      randomData[i] = Math.random();
    }
    let xData = tf.tensor1d(randomData);
    let yData = tf.add(tf.mul(xData,originA),originB);
    
    let A = tf.variable(tf.zeros([1]));
    let B = tf.variable(tf.zeros([1]));

    //y = A * x + B
    let y = x => tf.add(tf.mul(A,x),B);
    let loss = (y,yd) => y.sub(yd).square().mean();
    let optimizer = tf.train.sgd(0.01);
    console.log(`step: 0, A: ${A.dataSync()}, B: ${B.dataSync()}`);
    for (let i = 1; i <= 5000; i++) {
      optimizer.minimize(() => loss(y(xData),yData))
      if(i % 100 == 0){
        console.log(`step: ${i}, A: ${A.dataSync()}, B: ${B.dataSync()}`);
      }
    }

    console.log("======== TRAINING END ========");

    console.log(`Training result : A: ${A.dataSync()}, B: ${B.dataSync()}`);
    console.log(`Correct answer  : A: ${originA.dataSync()}, B: ${originB.dataSync()}`)
    
    //memory release
    originA.dispose();
    originB.dispose();
    xData.dispose();
    yData.dispose();
    A.dispose();
    B.dispose();
    console.log("========== PROG1 END ==========");
  })      
});
