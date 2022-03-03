/******************************************************************************
  Reads the distance something is in front of LIDAR and prints it to the Serial port

  Priyanka Makin @ SparkX Labs
  Original Creation Date: Sept 30, 2019

  This code is Lemonadeware; if you see me (or any other SparkFun employee) at the
  local, and you've found our code helpful, please buy us a round!

  Hardware Connections:
  Plug Qwiic LIDAR into Qwiic RedBoard using Qwiic cable.
  Set serial monitor to 115200 baud.

  Distributed as-is; no warranty is given.
******************************************************************************/
#include "LIDARLite_v4LED.h"


LIDARLite_v4LED myLIDAR; //Click here to get the library: http://librarymanager/All#SparkFun_LIDARLitev4 by SparkFun

void setup() {
  //manually reset board
  digitalWrite(4, HIGH);
  delay(200); 
  pinMode(4, OUTPUT);
  Serial.begin(115200);
  Serial.println("Qwiic LIDARLite_v4 examples");
  Wire.begin(); //Join I2C bus

  //check if LIDAR will acknowledge over I2C
  if (myLIDAR.begin() == false) {
    Serial.println("Device did not acknowledge! Freezing.");
    while(1);
  }
  Serial.println("LIDAR acknowledged!");
  //pinMode(LED_BUILTIN, OUTPUT);
  pinMode(7, OUTPUT);
}

void loop() {
  float newDistance;

  //getDistance() returns the distance reading in cm
  newDistance = myLIDAR.getDistance();

  //Print to Serial port
  Serial.print("New distance: ");
  Serial.print(newDistance/100);
  Serial.println(" m");

  delay(20);  //Don't hammer too hard on the I2C bus




  if (newDistance/100 < 2.05) {
    digitalWrite(7, HIGH);
  } else {
    digitalWrite(7, LOW);
  }
  delay(20);
//  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
//  delay(1000);                       // wait for a second
//  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//  delay(1000);                       // wait for a second
}
