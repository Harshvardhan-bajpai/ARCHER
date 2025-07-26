#include <WiFi.h>
#include <ESPAsyncWebServer.h>
#include "esp_camera.h"
#include <ESP32Servo.h>

const char* ssid = "harsh";
const char* password = "750808ba";

Servo gimbalServo;
const int servoPin = 14;
int currentAngle = 90;
const int minAngle = 45;
const int maxAngle = 135;

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27
#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      21
#define Y4_GPIO_NUM      19
#define Y3_GPIO_NUM      18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

AsyncWebServer server(80);

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }
}

void moveServo(String cmd) {
  if (cmd == "U") {
    currentAngle = max(minAngle, currentAngle - 5);
  } else if (cmd == "D") {
    currentAngle = min(maxAngle, currentAngle + 5);
  }
  gimbalServo.write(currentAngle);
  Serial.printf("Moved servo to %d° [%s]\n", currentAngle, cmd.c_str());
}

void setup() {
  Serial.begin(115200);
  delay(100);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.printf("ESP32-CAM IP: %s\n", WiFi.localIP().toString().c_str());

  setupCamera();

  gimbalServo.attach(servoPin);
  gimbalServo.write(currentAngle);

  server.on("/stream", HTTP_GET, [](AsyncWebServerRequest *request){
    AsyncWebServerResponse *response = request->beginChunkedResponse(
      "multipart/x-mixed-replace; boundary=frame",
      [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
        static camera_fb_t * fb = nullptr;
        static size_t pos = 0;
        static size_t header_len = 0;
        static char *header_buf = nullptr;

        if (!fb) {
          if (header_buf) {
            free(header_buf);
            header_buf = nullptr;
          }
          fb = esp_camera_fb_get();
          if (!fb) {
            return 0;
          }
          const char *header = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ";
          int len = snprintf(nullptr, 0, "%s%d\r\n\r\n", header, fb->len);
          header_buf = (char*)malloc(len + 1);
          snprintf(header_buf, len + 1, "%s%d\r\n\r\n", header, fb->len);
          header_len = len;
          pos = 0;
        }

        size_t bytes_to_send = 0;

        if (pos < header_len) {
          bytes_to_send = min(maxLen, header_len - pos);
          memcpy(buffer, header_buf + pos, bytes_to_send);
          pos += bytes_to_send;
          return bytes_to_send;
        }

        size_t fb_pos = pos - header_len;
        if (fb_pos < fb->len) {
          bytes_to_send = min(maxLen, fb->len - fb_pos);
          memcpy(buffer, fb->buf + fb_pos, bytes_to_send);
          pos += bytes_to_send;
          return bytes_to_send;
        }

        size_t tail_pos = pos - header_len - fb->len;
        if (tail_pos < 2) {
          const char *tail = "\r\n";
          bytes_to_send = min(maxLen, 2 - tail_pos);
          memcpy(buffer, tail + tail_pos, bytes_to_send);
          pos += bytes_to_send;
          return bytes_to_send;
        }

        esp_camera_fb_return(fb);
        fb = nullptr;
        pos = 0;

        return 0;
      }
    );
    request->send(response);
  });

  server.on("/cmd", HTTP_POST, [](AsyncWebServerRequest *request){
    if (request->hasParam("value", true)) {
      String val = request->getParam("value", true)->value();
      moveServo(val);
    }
    request->send(200, "text/plain", "OK");
  });

  server.begin();
}

void loop() {}
