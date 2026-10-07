# Báo cáo Day 6: Phát hiện calibration LiDAR-camera bị lệch

> Thay **mọi** ô có chữ ĐIỀN nằm trong ngoặc vuông bằng nội dung của bạn, xoá luôn cả dấu ngoặc vuông. Lệnh `python tools/check_submission.py` sẽ báo FAIL nếu còn sót bất kỳ chỗ nào.

- **Họ tên:** Hoàng Quốc Việt
- **MSSV:** 2A202602563
- **Lớp:** AI20K-T4
- **Link repo:** https://github.com/Catnip-harvest/HoangQuocViet-2A202602563-Track4-Day21
- **Topic:** A — LiDAR-camera projection QA (calibration)
- **Dataset:** data/kitti_mini (chính), data/nuscenes_mini_subset (so sánh, lỗi thời gian), data/synthetic (debug + tìm lỗi cài sẵn)
- **Các frame đã dùng:** 000008, 000011, 000049 (thí nghiệm chính); 000004, 000019 (demo khoảng cách); cả 20 frame KITTI và 80 keyframe nuScenes (so sánh metric, B5)

> Hãy viết ngắn: mỗi mục từ 3 đến 8 dòng, ưu tiên số liệu và hình ảnh.

## 1. Claim

Claim nháp (CP1): Lệch yaw 1° làm tỉ lệ điểm LiDAR của người đi bộ rơi đúng vào 2D box giảm hơn 20 điểm phần trăm, trong khi với xe con chỉ giảm dưới 5 điểm phần trăm.

## 2. Evidence

Bảng hoặc plot số liệu, kèm ảnh/video demo. Ghi rõ đường dẫn file trong `results/`.

| Cấu hình / mức perturb | Metric 1 | Metric 2 | Ghi chú |
|---|---|---|---|
| [ĐIỀN] | | | |

![demo](../results/figures/[ĐIỀN].png)

## 3. Failure case

Nêu khi nào hệ thống hoặc phương pháp fail, vì sao fail, và liên hệ tới lớp nào trong 6 lớp debug: I/O, Geometry, Time, Preprocess, Model, Metric.

![failure](../results/figures/fail_[ĐIỀN].png)

[ĐIỀN]

## 4. Khuyến nghị nếu triển khai thật

Use-case cụ thể (ADAS / robot / drone), trade-off và bước tiếp theo.

[ĐIỀN]

## 5. Cách chạy lại

Các lệnh tái tạo lại toàn bộ kết quả từ repo sạch.

```bash
[ĐIỀN]
```

## 6. Khai báo sử dụng AI

Ghi rõ đã dùng công cụ AI nào, dùng vào việc gì, và bạn đã tự kiểm chứng kết quả đó bằng cách nào. Nếu không dùng AI, ghi "Không sử dụng". Xem quy định ở `RULES.md` mục 2.

| Công cụ | Dùng cho việc gì | Bạn đã kiểm chứng thế nào |
|---|---|---|
| [ĐIỀN] | | |
