# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `KX-DAY01-report.zip`. Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
- Record hạng 1: `class_id=468`, `class_name=cab`, `rank=1`, `score=0.510915`, `taxonomy_name=ImageNet-1K` (`sample_id=traffic`).
- Đối chiếu `visuals/classification_top5.png`: ảnh gốc là một cảnh giao thông đông đúc, có nhiều xe buýt và ô tô. Năm lớp đứng đầu lần lượt là `cab` (0.510915), `minibus` (0.164284), `police_van` (0.085848), `recreational_vehicle` (0.054110) và `streetcar` (0.048193). Nhãn `cab` mô tả một khả năng nổi bật mà model suy ra cho toàn ảnh, nhưng không mô tả đầy đủ toàn cảnh hoặc liệt kê mọi phương tiện.
- Class list do taxonomy/dataset dùng để huấn luyện checkpoint quy định; trong trường hợp này checkpoint `yolo11n-cls.pt` sử dụng danh sách lớp ImageNet-1K. Người xây dựng dự án có thể chọn checkpoint/taxonomy phù hợp, nhưng annotator không tự thêm lớp ngoài class list của checkpoint.
- Cần giữ `class_id` để định danh ổn định khi xử lý máy, `class_name` để con người đọc và kiểm tra, và `taxonomy_name` để biết ID/tên đó thuộc bộ lớp nào, tránh nhầm khi hai taxonomy dùng ID hoặc tên khác nhau.
- Khi ảnh có nhiều chủ thể, guideline phải nêu rõ bài toán là single-label hay multi-label; nếu single-label thì quy định tiêu chí chọn nhãn chính (ví dụ chủ thể trung tâm/lớn nhất hoặc chủ đề của ảnh), cách xử lý khi không có chủ thể trội, và không suy đoán lớp không đủ bằng chứng. Nếu cần ghi từng vật thể thì phải chuyển sang detection/segmentation thay vì ép một nhãn cho toàn ảnh.
- Model score chỉ là độ tin cậy tương đối của checkpoint đối với các lớp ứng viên, không phải nhãn đúng do người xác nhận. Ground truth phải được tạo và kiểm tra theo guideline độc lập với prediction.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Record ('person', 'score': 0.912625, 'bbox_xyxy': [385.33, 69.24, 498.92, 348.92],'bbox_width': 113.58, 'bbox_height': 279.68)
- Box của person có score 0.912625 nằm ở vùng trung tâm lệch phải ảnh, với tọa độ từ (385.33, 69.24) đến (498.92, 348.92). Hai box oven nằm ở vùng trái và phải ảnh. Các box bowl và cup chủ yếu nằm ở vùng trái, phần thấp đến giữa ảnh, lệch về bên trái so với box person.   
- Ở threshold 0.35, model giữ 11 prediction. Khi tăng threshold lên 0.60, số prediction giảm còn 6 vì các box có score thấp hơn 0.60 bị loại. Threshold thấp giúp tăng độ bao phủ nhưng có thể tạo thêm box nhiễu; threshold cao giảm số box cần review nhưng có nguy cơ bỏ sót vật thể khó nhận diện.
- Với việc thăng threshold từ 0.35 lên 0.60 thì nó sẽ giữ các prediction có score cao hơn 0.60 và các box nhỏ hơn sẽ bị loại bỏ. Độ bao phủ sẽ bị giảm đi, số công việc của reviewer cũng giảm đi.
- Quy tắc box chặt: vẽ box sát theo phần nhìn thấy của đúng vật thể, bao phủ đủ vật thể nhưng không lấy thêm nền hoặc vật thể bên cạnh; mỗi instance được gán một box riêng
- Với object bị che khuất, cần quy định rõ box bao quanh phần nhìn thấy hay toàn bộ object được suy đoán. Với object bị cắt mép, cần quy định có gán nhãn hay không và cho phép box chạm biên ảnh. Nếu phần object nhìn thấy quá ít, bị che nghiêm trọng hoặc không xác định chắc class/instance, annotator phải đánh dấu escalation để reviewer quyết định.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Record (`instance_id`: 'kitchen-004', `class_name`: 'potted plant', `score`: 0.632235, `polygon_xy`: [[1.0, 2.0], [1.0, 148.0], [3.0, 148.0], [3.0, 147.0]])
- Polygon mô tả sát đường biên và hình dạng thực tế của từng instance, nên chính xác hơn bouding box hình chữ nhật. Nó cho biết rõ phần pixel thuộc về object, kể cả những phần có hình dạng không vuông vức, mỗi một instance đều có 1 polygon riêng
- `instance_id` là mã định danh cho một object cụ thể trong ảnh và nó ko phải 'class_id' đại diện cho 1 loại object
- Đề xuất một quy tắc biên mask:Polygon phải bám sát đường biên thực tế của object, chỉ bao phủ các pixel thuộc object và không lấy phần nền hoặc object khác. Mỗi instance có một polygon riêng.
- Guideline cần quy định cách xử lý vùng biên không rõ: có gán mask theo phần nhìn thấy hay suy đoán toàn bộ object, và object tiếp xúc nhau được tách ra như thế nào. Nếu không xác định chắc đường biên, class hoặc instance, annotator phải đánh dấu escalation để reviewer quyết định thay vì tự đoán.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho toàn ảnh hoặc nhiều nhãn nếu guideline quy định multi-label; lưu class ID, tên lớp và taxonomy. | Ảnh có nhiều chủ thể nhưng model chỉ trả một nhãn nổi bật; model score không phải ground truth. | Xác định nhãn theo nội dung ảnh và guideline, không chép máy score thành nhãn đúng; đánh dấu trường hợp không đủ bằng chứng. | Kiểm tra nhãn có phù hợp toàn ảnh, đúng class list/taxonomy và quy tắc single-label/multi-label hay không. |
| Phát hiện vật thể | Một record cho mỗi object, gồm class và bounding box `xyxy` theo pixel. | Box có thể rộng/hẹp chưa đúng, bỏ sót vật thể nhỏ hoặc khó thấy, và vật thể bị che khuất/cắt mép. | Gán một box riêng cho từng object, vẽ sát phần nhìn thấy, chọn class theo guideline và chuyển ca mơ hồ sang escalation. | Kiểm tra class, số lượng object, độ sát của box, box trùng/nhiễu và cách xử lý object bị che hoặc chạm biên ảnh. |
| Instance segmentation | Một record cho mỗi instance, gồm instance ID, class và polygon/mask theo pixel. | Đường biên mờ, object tiếp xúc hoặc bị che khiến khó tách polygon; polygon có thể ăn vào nền hoặc object khác. | Vẽ polygon bám biên phần nhìn thấy, tách từng instance riêng và đánh dấu vùng không chắc để reviewer quyết định. | Kiểm tra polygon có bám đúng biên, không chứa nền, tách đúng instance và nhất quán với guideline che khuất/tiếp xúc. |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: chỉ sử dụng ảnh và dữ liệu đúng phạm vi bài thực hành, không tự ý tải xuống, chia sẻ hoặc đưa dữ liệu nhạy cảm vào output/repository.
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho mentor/giảng viên phụ trách.

## 6. Danh sách bằng chứng

- [ ] `classification_predictions.json`
- [ ] `detection_predictions.json`
- [ ] `segmentation_predictions.json`
- [ ] `IMAGE_ATTRIBUTION.md`
- [ ] `visuals/classification_top5.png`
- [ ] `visuals/detection_predictions.png`
- [ ] `visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
