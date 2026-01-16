#include <SoftwareSerial.h>

SoftwareSerial BTserial(2, 3); // RX | TX

void setup()
{
   Serial.begin(38400);
   BTserial.begin(38400);
//   BTserial.begin(9600);
   Serial.println("Bluetooth send and receive test.");
}

void loop()
{
   if (Serial.available())
   {
      BTserial.write(Serial.read());
   }
   if (BTserial.available())
   {
      Serial.print(char(BTserial.read()));
   }
}