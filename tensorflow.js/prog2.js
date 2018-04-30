/* 
 * Find coefficients of quadratic function
 */

window.addEventListener("DOMContentLoaded", function() {
  document.getElementById("prog2").addEventListener("click",function(){
    
    console.log("==========  PROG2 START ==========");
    //y = 0.1 * x + 0.3 * x + 0.6
    const originA = tf.scalar(0.1);
    const originB = tf.scalar(0.3);
    const originC = tf.scalar(0.6);
    
    let randomData = [];
    for(let i=0;i<=100;i++){
      randomData[i] = Math.random();
    }
    let xData = tf.tensor1d(randomData);
    let yData = tf.add(tf.add(tf.mul(xData.square(),originA),tf.mul(xData,originB)),originC);
    
    let A = tf.variable(tf.zeros([1]));
    let B = tf.variable(tf.zeros([1]));
    let C = tf.variable(tf.zeros([1]));

    //y = A * x^2 + B * x + C
    let y = x => tf.add(tf.add(tf.mul(A,x.square()),tf.mul(B,x)),C);
    let loss = (y,yd) => y.sub(yd).square().mean();
    let optimizer = tf.train.sgd(0.5);
    console.log(`step: 0, A: ${A.dataSync()}, B: ${B.dataSync()}, C: ${C.dataSync()}`);
    for (let i = 1; i <= 5000; i++) {
      optimizer.minimize(() => loss(y(xData),yData))
      if(i % 100 == 0){
        console.log(`step: ${i}, A: ${A.dataSync()}, B: ${B.dataSync()}, C: ${C.dataSync()}`);
      }
    }
    console.log("======== TRAINING END ========");

    console.log(`Training result : A: ${A.dataSync()}, B: ${B.dataSync()}, C: ${C.dataSync()}`);
    console.log(`Correct answer  : A: ${originA.dataSync()}, B: ${originB.dataSync()}, C: ${originC.dataSync()}`)
  
    //memory release
    originA.dispose();
    originB.dispose();
    originC.dispose();
    xData.dispose();
    yData.dispose();
    A.dispose();
    B.dispose();
    C.dispose();
    
    console.log("========== PROG2 END ==========");
  })      
});
