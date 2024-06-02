## Crawl Lãi Suất Thị Trường
### Giới thiệu
Dự án này được thiết kế để thu thập và phân tích thông tin về lãi suất huy động từ các ngân hàng đối thủ và lãi suất được công bố bởi Ngân hàng Nhà nước Việt Nam. Mục tiêu của dự án là tạo một công cụ tracking thay đổi trong lãi suất huy động của thị trường, giúp các đội phân tích nhận diện và phản ứng nhanh với thị trường. Ngoài ra dữ liệu được crawl về hệ thống là một phần đầu vào của dự án forecast về các chỉ số tài chính của sản phẩm huy động vốn.

### Tính năng
- Crawl dữ liệu lãi suất từ các website của các ngân hàng đối thủ.
- Crawl dữ liệu lãi suất được công bố bởi Ngân hàng Nhà nước Việt Nam.
- Phân tích và so sánh lãi suất giữa các ngân hàng.
- Cung cấp dashboard trực quan hiển thị thông tin lãi suất.
- Gửi cảnh báo khi có thay đổi lãi suất.
- Truyền dữ liệu vào mô hình forecast chỉ số tài chính.

### Công nghệ sử dụng
- Python: Sử dụng để crawl dữ liệu và xử lý dữ liệu.
- BeautifulSoup: Thư viện để crawl dữ liệu từ các trang web.
- Pandas: Thư viện để phân tích và xử lý dữ liệu.
- Flask: Framework web để xây dựng API và dashboard.
- SQLite: Cơ sở dữ liệu để lưu trữ dữ liệu lãi suất.
