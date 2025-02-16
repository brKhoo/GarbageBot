#include <Arduino.h> // Include the Arduino library
long duration;
int distance;

const int trigPin = 10;
const int echoPin = 11;

void setup() {
 // put your setup code here
  pinMode(trigPin, OUTPUT)
  pinMODE(echoPin, INPUT)
  Serial.begin(115200);
}

void loop() {
  // put your main code here
  digitalWrite(trigPin,LOW);
  delayMicroseconds(2);

  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin,LOW);

  duration = pulseIn(echoPin, HIGH)
  distance = duration * (0.034/2);
  Serial.print('Distance: ')
  Serial.println(distance);
 
}