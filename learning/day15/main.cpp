/**
 * day15/main.cpp
 *
 * Day 15: YOLOv5 Object Detection Demo
 *
 * Demonstrates the complete YOLOv5 detection pipeline:
 *   1. Simulate YOLO network output tensors
 *   2. Apply sigmoid activation
 *   3. Decode bounding boxes
 *   4. Parse detections with confidence thresholding
 *   5. Run NMS (Non-Maximum Suppression)
 *   6. Scale coordinates back to original image
 *   7. Display results
 *
 * This is a standalone demo without OpenCV — the image processing
 * steps (letterbox, resize) are explained in comments.
 *
 * Note: Our Tensor<T> stores data as [C, H, W] internally.
 * For YOLO outputs [B, anchors, info, gridH, gridW], we flatten
 * dimensions to fit: C = B*anchors, H = info, W = gridH*gridW
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cassert>
#include <random>
#include <sstream>

#include "tensor.hpp"
#include "tensor_util.hpp"
#include "layer/layer.hpp"
#include "layer/layer_factory.hpp"
#include "layer/sigmoid.hpp"
#include "layer/relu.hpp"
#include "layer/flatten.hpp"
#include "yolo_detect.hpp"

using namespace learn_infer;

// COCO class names (first 20 for brevity)
static const std::vector<std::string> kCocoClassNames = {
    "person",       "bicycle",    "car",         "motorcycle",  "airplane",
    "bus",          "train",      "truck",       "boat",        "traffic light",
    "fire hydrant", "stop sign",  "parking meter", "bench",     "bird",
    "cat",          "dog",        "horse",       "sheep",       "cow"
    // ... (80 classes total in COCO)
};

// ─────────────────────────────────────────────────────
// Helper: print shape vector
// ─────────────────────────────────────────────────────
static void PrintShape(std::ostream& os, const std::vector<uint32_t>& s) {
  os << "[";
  for (uint32_t i = 0; i < s.size(); i++) {
    if (i) os << ", ";
    os << s[i];
  }
  os << "]";
}

// ─────────────────────────────────────────────────────
// Tensor helper for YOLO data layout
//
// YOLO detection heads produce [batch, 3, 6+num_classes, grid, grid]
// We map this to our 3D tensor as:
//   C = batch * 3 (anchors)
//   H = 6 + num_classes (info per anchor)
//   W = grid * grid (spatial positions)
//
// Access pattern: at(anchor_idx, info_idx, grid_pos)
// where grid_pos = row * grid + col
// ─────────────────────────────────────────────────────

/**
 * Create a detection head tensor.
 * Shape: [batch * num_anchors, num_info, grid * grid]
 */
static std::shared_ptr<Tensor<float>> CreateHeadTensor(
    int batch, int num_anchors, int num_info, int grid) {
  uint32_t c = static_cast<uint32_t>(batch * num_anchors);
  uint32_t h = static_cast<uint32_t>(num_info);
  uint32_t w = static_cast<uint32_t>(grid * grid);
  return std::make_shared<Tensor<float>>(c, h, w);
}

/**
 * Get spatial grid size from tensor width.
 */
static int GetGridSize(const Tensor<float>* t) {
  uint32_t w = t->cols();
  // grid * grid = w, so grid = sqrt(w)
  int g = static_cast<int>(std::round(std::sqrt(static_cast<float>(w))));
  return g;
}

/**
 * Helper to access head tensor values.
 * head->at(anchor_idx, info_idx, row, col)
 */
static float HeadAt(const std::shared_ptr<Tensor<float>>& head, int anchor, int info, int row, int col) {
  int grid = GetGridSize(head.get());
  return head->at(static_cast<uint32_t>(anchor),
                  static_cast<uint32_t>(info),
                  static_cast<uint32_t>(row * grid + col));
}

static void HeadSet(std::shared_ptr<Tensor<float>> head, int anchor, int info, int row, int col, float val) {
  int grid = GetGridSize(head.get());
  head->at(static_cast<uint32_t>(anchor),
           static_cast<uint32_t>(info),
           static_cast<uint32_t>(row * grid + col)) = val;
}

// ─────────────────────────────────────────────────────
// Demo 1: Sigmoid Activation
// ─────────────────────────────────────────────────────
static void DemoSigmoid() {
  std::cout << "=== Demo 1: Sigmoid Activation ===\n";

  // Sigmoid is applied to YOLO raw outputs to constrain values to [0, 1]
  auto input = std::make_shared<Tensor<float>>(1, 1, 10);
  for (int i = 0; i < 10; i++) {
    input->at(0, 0, i) = (i - 5) * 0.5f;  // Values from -2.5 to +2.5
  }

  std::cout << "  Input:  ";
  for (int i = 0; i < 10; i++) {
    std::cout << std::fixed << std::setprecision(3) << input->at(0, 0, i) << " ";
  }
  std::cout << "\n";

  // Run sigmoid layer
  std::vector<STensor> in_tensors{input};
  std::vector<STensor> out_tensors{nullptr};
  auto sigmoid_layer = LayerRegisterer<float>::CreateLayer("nn.Sigmoid");
  sigmoid_layer->Forward(in_tensors, out_tensors);

  auto output = out_tensors[0];
  std::cout << "  Output: ";
  for (int i = 0; i < 10; i++) {
    std::cout << std::fixed << std::setprecision(3) << output->at(0, 0, i) << " ";
  }
  std::cout << "\n";

  // Verify: sigmoid(0) = 0.5
  std::cout << "  sigmoid(0) = " << std::setprecision(4) << output->at(0, 0, 5)
            << " (expected ~0.5)\n\n";
}

// ─────────────────────────────────────────────────────
// Demo 2: Flatten & View (Reshape)
// ─────────────────────────────────────────────────────
static void DemoFlattenView() {
  std::cout << "=== Demo 2: Flatten & View (Reshape) ===\n";

  // YOLO produces 3 detection heads with different grid sizes:
  //   Head 1: [1, 255, 80, 80]  (stride 8, small objects)
  //   Head 2: [1, 255, 40, 40]  (stride 16)
  //   Head 3: [1, 255, 20, 20]  (stride 32, large objects)
  // These are permuted and flattened into a single tensor

  // Simulate a small detection head: [1, 6, 16] = [batch, info, grid*grid]
  auto input = std::make_shared<Tensor<float>>(1, 6, 16);
  for (uint32_t i = 0; i < input->size(); i++) {
    input->index(i) = 1.0f;
  }

  // Flatten: collapse to 1D
  std::vector<STensor> in_t{input};
  std::vector<STensor> out_t{nullptr};
  auto flatten = LayerRegisterer<float>::CreateLayer("nn.Flatten");
  flatten->Forward(in_t, out_t);

  std::cout << "  Input shape:  ";
  PrintShape(std::cout, input->shapes());
  std::cout << "\n";
  std::cout << "  Flattened:    ";
  PrintShape(std::cout, out_t[0]->shapes());
  std::cout << "\n";

  // View: reshape to a specific target shape
  auto view_layer = std::make_shared<learn_infer::ViewLayer>(1, 24, 4);
  in_t[0] = input;
  out_t[0] = nullptr;
  view_layer->Forward(in_t, out_t);
  std::cout << "  View(24,4):   ";
  PrintShape(std::cout, out_t[0]->shapes());
  std::cout << "\n\n";
}

// ─────────────────────────────────────────────────────
// Demo 3: Simulated YOLO Detection Pipeline
// ─────────────────────────────────────────────────────
static void DemoYOLODetection() {
  std::cout << "=== Demo 3: YOLO Detection Pipeline ===\n";

  // ── Parameters ──
  const int num_classes = 20;  // Simplified COCO subset
  const int num_info = 6 + num_classes;  // 4 bbox + 1 obj + num_classes
  const int img_w = 640;
  const int img_h = 640;

  // YOLOv5 uses 3 anchor groups (one per detection scale)
  // P5 anchors (stride 32, large objects):
  const std::vector<float> anchors_p5 = {136, 40, 230, 37, 321, 78};
  // P4 anchors (stride 16, medium objects):
  const std::vector<float> anchors_p4 = {30, 61, 62, 45, 59, 119};
  // P3 anchors (stride 8, small objects):
  const std::vector<float> anchors_p3 = {10, 13, 16, 30, 33, 23};

  // ── Step 1: Simulate detection outputs ──
  std::cout << "  Simulating YOLO detection heads with small grids...\n";

  // Use tiny grids for demo: 5x5, 3x3, 2x2
  int grid1 = 5, grid2 = 3, grid3 = 2;

  auto head1 = CreateHeadTensor(1, 3, num_info, grid1);
  auto head2 = CreateHeadTensor(1, 3, num_info, grid2);
  auto head3 = CreateHeadTensor(1, 3, num_info, grid3);

  std::cout << "    Head1 (stride 8, grid " << grid1 << "x" << grid1 << "): ";
  PrintShape(std::cout, head1->shapes());
  std::cout << "\n";
  std::cout << "    Head2 (stride 16, grid " << grid2 << "x" << grid2 << "): ";
  PrintShape(std::cout, head2->shapes());
  std::cout << "\n";
  std::cout << "    Head3 (stride 32, grid " << grid3 << "x" << grid3 << "): ";
  PrintShape(std::cout, head3->shapes());
  std::cout << "\n";

  // Initialize all values to a negative bias so sigmoid(x) << 0.5
  // This simulates a real network where most anchors have low confidence
  for (uint64_t i = 0; i < head1->size(); i++) head1->index(i) = -3.0f;
  for (uint64_t i = 0; i < head2->size(); i++) head2->index(i) = -3.0f;
  for (uint64_t i = 0; i < head3->size(); i++) head3->index(i) = -3.0f;

  // Now inject some "detections" with high confidence at specific anchors
  // Detection 1: "person" at anchor(0), grid(2,2) of head1
  HeadSet(head1, 0, 0, 2, 2, 0.5f);   // cx (before sigmoid)
  HeadSet(head1, 0, 1, 2, 2, 0.5f);   // cy
  HeadSet(head1, 0, 2, 2, 2, 0.3f);   // w
  HeadSet(head1, 0, 3, 2, 2, 0.4f);   // h
  HeadSet(head1, 0, 4, 2, 2, 2.0f);   // obj_conf -> sigmoid(2) ~= 0.88
  HeadSet(head1, 0, 5, 2, 2, 1.5f);   // class 0 (person) -> sigmoid ~= 0.82

  // Detection 2: "car" at anchor(1), grid(1,1) of head1
  HeadSet(head1, 1, 0, 1, 1, 0.6f);
  HeadSet(head1, 1, 1, 1, 1, 0.4f);
  HeadSet(head1, 1, 2, 1, 1, 0.5f);
  HeadSet(head1, 1, 3, 1, 1, 0.3f);
  HeadSet(head1, 1, 4, 1, 1, 1.8f);   // obj_conf ~= 0.86
  HeadSet(head1, 1, 5 + 2, 1, 1, 1.6f);   // class 2 (car) ~= 0.83

  // Detection 3: "dog" at anchor(0), grid(0,0) of head3 (large object)
  HeadSet(head3, 0, 0, 0, 0, 0.5f);
  HeadSet(head3, 0, 1, 0, 0, 0.5f);
  HeadSet(head3, 0, 2, 0, 0, 0.4f);
  HeadSet(head3, 0, 3, 0, 0, 0.3f);
  HeadSet(head3, 0, 4, 0, 0, 1.5f);   // obj_conf ~= 0.82
  HeadSet(head3, 0, 5 + 16, 0, 0, 1.7f);  // class 16 (dog) ~= 0.85

  // Detection 4: Overlapping with detection 1 (to test NMS)
  HeadSet(head1, 2, 0, 2, 2, 0.52f);  // Very close box to detection 1
  HeadSet(head1, 2, 1, 2, 2, 0.48f);
  HeadSet(head1, 2, 2, 2, 2, 0.35f);
  HeadSet(head1, 2, 3, 2, 2, 0.42f);
  HeadSet(head1, 2, 4, 2, 2, 1.0f);   // obj_conf ~= 0.73
  HeadSet(head1, 2, 5, 2, 2, 1.2f);   // class 0 (person) ~= 0.77

  // Detection 5: "cat" at head2
  HeadSet(head2, 0, 0, 1, 1, 0.5f);
  HeadSet(head2, 0, 1, 1, 1, 0.5f);
  HeadSet(head2, 0, 2, 1, 1, 0.3f);
  HeadSet(head2, 0, 3, 1, 1, 0.3f);
  HeadSet(head2, 0, 4, 1, 1, 1.6f);   // obj_conf ~= 0.83
  HeadSet(head2, 0, 5 + 15, 1, 1, 1.4f);  // class 15 (cat) ~= 0.80

  // ── Step 2: Apply sigmoid to all heads ──
  std::cout << "\n  Step 2: Applying sigmoid activation to raw outputs...\n";

  auto sigmoid = LayerRegisterer<float>::CreateLayer("nn.Sigmoid");
  std::vector<STensor> sig_in;
  std::vector<STensor> sig_out{nullptr};

  sig_in = {head1};
  sigmoid->Forward(sig_in, sig_out);
  head1 = sig_out[0];

  sig_in = {head2};
  sig_out[0] = nullptr;
  sigmoid->Forward(sig_in, sig_out);
  head2 = sig_out[0];

  sig_in = {head3};
  sig_out[0] = nullptr;
  sigmoid->Forward(sig_in, sig_out);
  head3 = sig_out[0];

  std::cout << "  Sigmoid applied. All values now in [0, 1].\n";
  std::cout << "  Example — detection 1 obj_conf: "
            << std::fixed << std::setprecision(4)
            << HeadAt(head1, 0, 4, 2, 2)
            << " (sigmoid(2.0) = 0.8808)\n";

  // ── Step 3: Decode bounding boxes for each head ──
  std::cout << "\n  Step 3: Decoding bounding boxes per detection head...\n";

  // Decode head 1 (stride 8, grid 5x5)
  {
    const auto& anchors = anchors_p3;
    int stride = 8;
    int grid = grid1;
    for (int a = 0; a < 3; a++) {
      for (int i = 0; i < grid; i++) {
        for (int j = 0; j < grid; j++) {
          float obj_conf = HeadAt(head1, a, 4, i, j);
          if (obj_conf > 0.5f) {
            float cx = HeadAt(head1, a, 0, i, j);
            float cy = HeadAt(head1, a, 1, i, j);
            float w = HeadAt(head1, a, 2, i, j);
            float h = HeadAt(head1, a, 3, i, j);

            std::vector<float> decoded(4);
            DecodeBox(cx, cy, w, h,
                      grid, grid,
                      anchors[a * 2], anchors[a * 2 + 1],
                      stride,
                      img_w, img_h,
                      decoded);

            HeadSet(head1, a, 0, i, j, decoded[0]);
            HeadSet(head1, a, 1, i, j, decoded[1]);
            HeadSet(head1, a, 2, i, j, decoded[2]);
            HeadSet(head1, a, 3, i, j, decoded[3]);

            std::cout << "    Head1 anchor[" << a << "] grid[" << i << "][" << j
                      << "]: box=(" << std::fixed << std::setprecision(1)
                      << decoded[0] << ", " << decoded[1]
                      << ", " << decoded[2] << "x" << decoded[3] << ")\n";
          }
        }
      }
    }
  }

  // Decode head 2 (stride 16, grid 3x3)
  {
    const auto& anchors = anchors_p4;
    int stride = 16;
    int grid = grid2;
    for (int a = 0; a < 3; a++) {
      for (int i = 0; i < grid; i++) {
        for (int j = 0; j < grid; j++) {
          float obj_conf = HeadAt(head2, a, 4, i, j);
          if (obj_conf > 0.5f) {
            float cx = HeadAt(head2, a, 0, i, j);
            float cy = HeadAt(head2, a, 1, i, j);
            float w = HeadAt(head2, a, 2, i, j);
            float h = HeadAt(head2, a, 3, i, j);

            std::vector<float> decoded(4);
            DecodeBox(cx, cy, w, h,
                      grid, grid,
                      anchors[a * 2], anchors[a * 2 + 1],
                      stride,
                      img_w, img_h,
                      decoded);

            HeadSet(head2, a, 0, i, j, decoded[0]);
            HeadSet(head2, a, 1, i, j, decoded[1]);
            HeadSet(head2, a, 2, i, j, decoded[2]);
            HeadSet(head2, a, 3, i, j, decoded[3]);

            std::cout << "    Head2 anchor[" << a << "] grid[" << i << "][" << j
                      << "]: box=(" << std::fixed << std::setprecision(1)
                      << decoded[0] << ", " << decoded[1]
                      << ", " << decoded[2] << "x" << decoded[3] << ")\n";
          }
        }
      }
    }
  }

  // Decode head 3 (stride 32, grid 2x2)
  {
    const auto& anchors = anchors_p5;
    int stride = 32;
    int grid = grid3;
    for (int a = 0; a < 3; a++) {
      for (int i = 0; i < grid; i++) {
        for (int j = 0; j < grid; j++) {
          float obj_conf = HeadAt(head3, a, 4, i, j);
          if (obj_conf > 0.5f) {
            float cx = HeadAt(head3, a, 0, i, j);
            float cy = HeadAt(head3, a, 1, i, j);
            float w = HeadAt(head3, a, 2, i, j);
            float h = HeadAt(head3, a, 3, i, j);

            std::vector<float> decoded(4);
            DecodeBox(cx, cy, w, h,
                      grid, grid,
                      anchors[a * 2], anchors[a * 2 + 1],
                      stride,
                      img_w, img_h,
                      decoded);

            HeadSet(head3, a, 0, i, j, decoded[0]);
            HeadSet(head3, a, 1, i, j, decoded[1]);
            HeadSet(head3, a, 2, i, j, decoded[2]);
            HeadSet(head3, a, 3, i, j, decoded[3]);

            std::cout << "    Head3 anchor[" << a << "] grid[" << i << "][" << j
                      << "]: box=(" << std::fixed << std::setprecision(1)
                      << decoded[0] << ", " << decoded[1]
                      << ", " << decoded[2] << "x" << decoded[3] << ")\n";
          }
        }
      }
    }
  }

  // ── Step 4: Flatten all heads into one detection list ──
  std::cout << "\n  Step 4: Collecting all detections from all heads...\n";

  int total_elements = 3 * grid1 * grid1 + 3 * grid2 * grid2 + 3 * grid3 * grid3;
  std::cout << "  Total anchor positions: " << total_elements << "\n";

  // ── Step 5: Parse detections ──
  const float conf_threshold = 0.3f;
  std::cout << "\n  Step 5: Parsing detections (conf > " << conf_threshold << ")...\n";

  // Collect manually from each head
  std::vector<Detection> raw_detections;
  auto CollectDetections = [&](std::shared_ptr<Tensor<float>> head, int stride,
                                int grid, int batch) {
    for (int b = 0; b < batch; b++) {
      for (int a = 0; a < 3; a++) {
        for (int i = 0; i < grid; i++) {
          for (int j = 0; j < grid; j++) {
            float obj_conf = HeadAt(head, a, 4, i, j);
            if (obj_conf < conf_threshold) continue;

            // Find best class
            float best_cls_prob = -1.0f;
            int best_cls_id = 0;
            for (int c = 0; c < num_classes; c++) {
              float cls_prob = HeadAt(head, a, 5 + c, i, j);
              if (cls_prob > best_cls_prob) {
                best_cls_prob = cls_prob;
                best_cls_id = c;
              }
            }

            float confidence = obj_conf * best_cls_prob;
            if (confidence < conf_threshold) continue;

            // Filter out degenerate boxes (zero or near-zero width/height)
            int w = static_cast<int>(HeadAt(head, a, 2, i, j));
            int h = static_cast<int>(HeadAt(head, a, 3, i, j));
            if (w <= 0 || h <= 0) continue;

            int cx = static_cast<int>(HeadAt(head, a, 0, i, j));
            int cy = static_cast<int>(HeadAt(head, a, 1, i, j));

            Detection det;
            det.width = w;
            det.height = h;
            det.x = cx - w / 2;
            det.y = cy - h / 2;
            det.confidence = confidence;
            det.class_id = best_cls_id;
            raw_detections.push_back(det);
          }
        }
      }
    }
  };

  CollectDetections(head1, 8, grid1, 1);
  CollectDetections(head2, 16, grid2, 1);
  CollectDetections(head3, 32, grid3, 1);

  std::cout << "  Raw detections before NMS: " << raw_detections.size() << "\n";
  PrintDetections(raw_detections, &kCocoClassNames);

  // ── Step 6: NMS ──
  std::cout << "\n  Step 6: Running NMS (IoU threshold = 0.45)...\n";

  auto final_detections = NMS(raw_detections, 0.45f);
  std::cout << "  Detections after NMS: " << final_detections.size() << "\n";
  PrintDetections(final_detections, &kCocoClassNames);

  // ── Step 7: Coordinate scaling (letterbox reverse transform) ──
  std::cout << "\n  Step 7: Scaling coordinates to original image size...\n";

  // Simulate letterbox params: original image 480x360, scaled to 640x640
  int orig_w = 480, orig_h = 360;
  float gain = std::min(static_cast<float>(img_w) / orig_w,
                        static_cast<float>(img_h) / orig_h);
  int new_w = static_cast<int>(orig_w * gain);
  int new_h = static_cast<int>(orig_h * gain);
  int pad_x = (img_w - new_w) / 2;
  int pad_y = (img_h - new_h) / 2;

  std::cout << "  Original image: " << orig_w << "x" << orig_h << "\n";
  std::cout << "  Scaled image:   " << new_w << "x" << new_h
            << " (padded to " << img_w << "x" << img_h << ")\n";
  std::cout << "  Gain: " << std::fixed << std::setprecision(3) << gain
            << ", Pad: (" << pad_x << ", " << pad_y << ")\n";

  for (auto& det : final_detections) {
    std::vector<float> box{
        static_cast<float>(det.x),
        static_cast<float>(det.y),
        static_cast<float>(det.width),
        static_cast<float>(det.height)
    };
    ScaleBox(box, img_w, img_h, orig_w, orig_h, gain, pad_x, pad_y);

    det.x = static_cast<int>(box[0]);
    det.y = static_cast<int>(box[1]);
    det.width = static_cast<int>(box[2]);
    det.height = static_cast<int>(box[3]);
  }

  std::cout << "\n  Final detections on original image:\n";
  PrintDetections(final_detections, &kCocoClassNames);
}

// ─────────────────────────────────────────────────────
// Demo 4: NMS in Detail
// ─────────────────────────────────────────────────────
static void DemoNMS() {
  std::cout << "=== Demo 4: NMS (Non-Maximum Suppression) Detail ===\n";

  // Create overlapping boxes manually
  std::vector<Detection> boxes = {
      // Two highly overlapping "person" detections
      {100, 150, 80, 120, 0.92f, 0},
      {105, 155, 75, 115, 0.85f, 0},
      // A non-overlapping "car" detection
      {300, 200, 100, 60, 0.78f, 2},
      // Another overlapping box (should be suppressed)
      {95, 148, 85, 125, 0.70f, 0},
      // A far-away "dog" detection
      {450, 350, 50, 70, 0.65f, 16},
      // Yet another overlapping with the person boxes
      {110, 160, 70, 110, 0.60f, 0},
  };

  std::cout << "  Input: " << boxes.size() << " detections\n";
  for (size_t i = 0; i < boxes.size(); i++) {
    auto& b = boxes[i];
    std::cout << "    Box " << i << ": (" << b.x << "," << b.y << ","
              << b.width << "x" << b.height << ") conf="
              << std::fixed << std::setprecision(2) << b.confidence
              << " class=" << kCocoClassNames[b.class_id] << "\n";
  }

  // Compute pairwise IoU for the first two boxes
  std::vector<float> box0{static_cast<float>(boxes[0].x), static_cast<float>(boxes[0].y),
                           static_cast<float>(boxes[0].width), static_cast<float>(boxes[0].height)};
  std::vector<float> box1{static_cast<float>(boxes[1].x), static_cast<float>(boxes[1].y),
                           static_cast<float>(boxes[1].width), static_cast<float>(boxes[1].height)};
  float iou = ComputeIoU(box0, box1);
  std::cout << "  IoU(Box 0, Box 1) = " << std::setprecision(3) << iou << "\n";

  auto result = NMS(boxes, 0.45f);

  std::cout << "\n  After NMS: " << result.size() << " detections\n";
  for (const auto& b : result) {
    std::cout << "    (" << b.x << "," << b.y << ","
              << b.width << "x" << b.height << ") conf="
              << std::setprecision(2) << b.confidence
              << " class=" << kCocoClassNames[b.class_id] << "\n";
  }
  std::cout << "\n  (Suppressed " << (boxes.size() - result.size())
            << " overlapping person detections, kept car and dog)\n\n";
}

// ─────────────────────────────────────────────────────
// Main entry point
// ─────────────────────────────────────────────────────
int main() {
  std::cout << "================================================================\n";
  std::cout << "  Day 15: YOLOv5 Object Detection Demo\n";
  std::cout << "================================================================\n";
  std::cout << "  Topics:\n";
  std::cout << "    - Sigmoid activation for YOLO outputs\n";
  std::cout << "    - Flatten/View (reshape) layers\n";
  std::cout << "    - Bounding box decoding\n";
  std::cout << "    - Confidence thresholding\n";
  std::cout << "    - NMS (Non-Maximum Suppression)\n";
  std::cout << "    - Coordinate scaling (letterbox reverse)\n";
  std::cout << "================================================================\n\n";

  DemoSigmoid();
  DemoFlattenView();
  DemoYOLODetection();
  DemoNMS();

  std::cout << "All Day 15 demos completed successfully!\n";
  return 0;
}
