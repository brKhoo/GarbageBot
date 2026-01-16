#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position

void setup() {
  Serial.begin(38400);
  myservo.attach(4);  // attaches the servo on pin 9 to the servo object
}

void direction(int dir) {
  myservo.writeMicroseconds(1500);
  delay(500);
  myservo.writeMicroseconds(1500 + (dir * 200));
  delay(500);
  myservo.writeMicroseconds(1500);
  delay(500);
}

void loop() {
  direction(1);
  direction(-1);

  
  // SERVO CONTROL SERIAL
  //  while(!Serial.available());
  //
  //  int next = Serial.parseInt();
  //
  //  Serial.print("got int ");
  //  Serial.println(next);
  //
  //  myservo.write(next);
  
  // SERVO SWEEP
  //for (pos = 0; pos <= 180; pos += 1) { // goes from 0 degrees to 180 degrees
  //  // in steps of 1 degree
  //  myservo.write(pos);              // tell servo to go to position in variable 'pos'
  //  delay(15);                       // waits 15ms for the servo to reach the position
  //}
  //for (pos = 180; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
  //  myservo.write(pos);              // tell servo to go to position in variable 'pos'
  //  delay(15);                       // waits 15ms for the servo to reach the position
  //}


}