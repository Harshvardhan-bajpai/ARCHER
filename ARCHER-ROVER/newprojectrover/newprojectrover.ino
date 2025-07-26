#include <Servo.h>

#define ENA 2
#define IN1 3
#define IN2 4

#define IN3 5
#define IN4 6
#define ENB 7

#define RC_CH1 8
#define RC_CH2 9

#define RC_TIMEOUT 30000
#define CENTER_POSITION 1500
#define DEADZONE 50

#define MIN_SPEED 50
#define MAX_SPEED 255
#define MIN_SIGNAL 1000
#define MAX_SIGNAL 2000
#define SIGNAL_TIMEOUT 500

Servo gimbalServo;
Servo fireServo;
#define GIMBAL_PIN 10
#define FIRE_PIN 11

int lastValidThrottle = CENTER_POSITION;
int lastValidSteering = CENTER_POSITION;
unsigned long lastSignalTime = 0;
int currentGimbalAngle = 90;

void setup() {
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  pinMode(RC_CH1, INPUT); pinMode(RC_CH2, INPUT);

  Serial.begin(9600);
  
  Serial.println("Tank Drive RC + Bluetooth + Servo Control Ready");

  gimbalServo.attach(GIMBAL_PIN);
  fireServo.attach(FIRE_PIN);
  gimbalServo.write(currentGimbalAngle);
  fireServo.write(30);

  stopMotors();
}

void loop() {
  if (Serial.available()) {
    processCommand(Serial.readStringUntil('\n'));
    return;
  }
  
  int throttle = pulseIn(RC_CH1, HIGH, RC_TIMEOUT);
  int steering = pulseIn(RC_CH2, HIGH, RC_TIMEOUT);

  if (isValidSignal(throttle)) {
    lastValidThrottle = throttle;
    lastSignalTime = millis();
  } else if (millis() - lastSignalTime < SIGNAL_TIMEOUT) {
    throttle = lastValidThrottle;
  } else {
    throttle = CENTER_POSITION;
  }

  if (isValidSignal(steering)) {
    lastValidSteering = steering;
    lastSignalTime = millis();
  } else if (millis() - lastSignalTime < SIGNAL_TIMEOUT) {
    steering = lastValidSteering;
  } else {
    steering = CENTER_POSITION;
  }

  int throttleOffset = throttle - CENTER_POSITION;
  int steeringOffset = steering - CENTER_POSITION;

  if (abs(throttleOffset) < DEADZONE) throttleOffset = 0;
  if (abs(steeringOffset) < DEADZONE) steeringOffset = 0;

  int leftPower = throttleOffset + steeringOffset;
  int rightPower = throttleOffset - steeringOffset;

  leftPower = constrain(leftPower, -500, 500);
  rightPower = constrain(rightPower, -500, 500);

  controlMotor(leftPower, IN1, IN2, ENA);
  controlMotor(rightPower, IN3, IN4, ENB);

  delay(20);
}

void processCommand(String input) {
  input.trim();

  if (input.startsWith("G:")) {
    int angle = input.substring(2).toInt();
    currentGimbalAngle = constrain(angle, 70, 140);
    gimbalServo.write(currentGimbalAngle);
  } else if (input == "G") {
    currentGimbalAngle = constrain(currentGimbalAngle - 5, 70, 140);
    gimbalServo.write(currentGimbalAngle);
  } else if (input == "H") {
    currentGimbalAngle = constrain(currentGimbalAngle + 5, 70, 140);
    gimbalServo.write(currentGimbalAngle);
  } else if (input == "F") {
    fireServo.write(160);
    delay(1000);
    fireServo.write(30);
  } else {
    if (input == "W") {
      controlMotor(300, IN1, IN2, ENA);
      controlMotor(300, IN3, IN4, ENB);
    } else if (input == "S") {
      controlMotor(-300, IN1, IN2, ENA);
      controlMotor(-300, IN3, IN4, ENB);
    } else if (input == "D") {
      controlMotor(-300, IN1, IN2, ENA);
      controlMotor(300, IN3, IN4, ENB);
    } else if (input == "A") {
      controlMotor(300, IN1, IN2, ENA);
      controlMotor(-300, IN3, IN4, ENB);
    } else if (input == "X") {
      stopMotors();
    }
  }
}

bool isValidSignal(int signal) {
  return (signal >= MIN_SIGNAL && signal <= MAX_SIGNAL);
}

void controlMotor(int power, int in1, int in2, int enPin) {
  int pwm = 0;

  if (power > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    pwm = map(power, 0, 500, MIN_SPEED, MAX_SPEED);
  } else if (power < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    pwm = map(-power, 0, 500, MIN_SPEED, MAX_SPEED);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    pwm = 0;
  }

  analogWrite(enPin, pwm);
}

void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}