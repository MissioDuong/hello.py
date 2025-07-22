import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import warnings
import ollama  # Added import for local LLM chat
warnings.filterwarnings('ignore')

# Suppress Streamlit warnings
import logging
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Cấu hình trang
st.set_page_config(
    page_title="AI Supply Chain Dashboard - ABC Company",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
        background-color: #fef3c7;
        color: black;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
        background-color: #d1fae5;
        color: black;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #ef4444;
        background-color: #fee2e2;
        color: black;
    }
    .notification-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
        background-color: #dbeafe;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        color: black;
    }
    .solution-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 2px solid #e5e7eb;
        background-color: #f9fafb;
        transition: all 0.3s ease;
        color: black;
    }
    .solution-card:hover {
        border-color: #3b82f6;
        background-color: #eff6ff;
    }
    .solution-card.best {
        border-color: #10b981;
        background-color: #ecfdf5;
    }
    .solution-card.medium {
        border-color: #f59e0b;
        background-color: #fffbeb;
    }
    .solution-card.urgent {
        border-color: #ef4444;
        background-color: #fef2f2;
    }
    .timer-box {
        padding: 0.5rem;
        border-radius: 0.25rem;
        background-color: #fef3c7;
        border: 1px solid #f59e0b;
        text-align: center;
        font-weight: bold;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if 'simulation_running' not in st.session_state:
    st.session_state.simulation_running = False
if 'simulation_step' not in st.session_state:
    st.session_state.simulation_step = 0
if 'inventory_data' not in st.session_state:
    st.session_state.inventory_data = {
        'Pin lithium': {'current': 2500, 'required': 3500, 'trend': -50},
        'Motor điện': {'current': 1800, 'required': 2200, 'trend': -30},
        'Khung xe': {'current': 3200, 'required': 2800, 'trend': -40},
        'Bánh xe': {'current': 4500, 'required': 4200, 'trend': -60},
        'Phanh đĩa': {'current': 2100, 'required': 2800, 'trend': -35}
    }
if 'route_data' not in st.session_state:
    st.session_state.route_data = {
        'Tuyến A (Trung Quốc)': {'risk': 85, 'delay': '7-10 ngày', 'position': 0, 'speed': 2},
        'Tuyến B (Đài Loan)': {'risk': 25, 'delay': '0-2 ngày', 'position': 0, 'speed': 3},
        'Tuyến C (Nhật Bản)': {'risk': 45, 'delay': '3-5 ngày', 'position': 0, 'speed': 2.5}
    }
if 'orders_processed' not in st.session_state:
    st.session_state.orders_processed = 0
if 'ai_decisions' not in st.session_state:
    st.session_state.ai_decisions = []
if 'active_notifications' not in st.session_state:
    st.session_state.active_notifications = []
if 'notification_history' not in st.session_state:
    st.session_state.notification_history = []
if 'notification_chats' not in st.session_state:
    st.session_state.notification_chats = {}

# Lớp AI Decision Engine với Notification System
class AIDecisionEngine:
    def __init__(self):
        self.risk_threshold = 70
        self.inventory_threshold = 0.8
        self.decision_history = []
        self.notification_id_counter = 0
    
    def generate_notification_id(self):
        """Tạo ID duy nhất cho notification"""
        self.notification_id_counter += 1
        return f"NOTIF_{self.notification_id_counter}_{int(time.time())}"
    
    def analyze_inventory_risk(self, inventory_data):
        """Phân tích rủi ro tồn kho"""
        risk_items = []
        try:
            for item, data in inventory_data.items():
                # Fix division by zero error
                if data['required'] <= 0:
                    continue
                ratio = data['current'] / data['required']
                if ratio < self.inventory_threshold:
                    risk_level = 'HIGH' if ratio < 0.7 else 'MEDIUM'
                    risk_items.append({
                        'item': item,
                        'ratio': ratio,
                        'risk': risk_level,
                        'shortage': max(0, data['required'] - data['current'])
                    })
        except Exception as e:
            st.error(f"Lỗi phân tích tồn kho: {str(e)}")
        return risk_items
    
    def analyze_route_risk(self, route_data):
        """Phân tích rủi ro tuyến vận chuyển"""
        high_risk_routes = []
        try:
            for route, data in route_data.items():
                if data['risk'] >= self.risk_threshold:
                    high_risk_routes.append({
                        'route': route,
                        'risk': data['risk'],
                        'delay': data['delay']
                    })
        except Exception as e:
            st.error(f"Lỗi phân tích tuyến vận chuyển: {str(e)}")
        return high_risk_routes
    
    def predict_demand(self, current_orders, trend_factor=1.4):
        """Dự báo nhu cầu sử dụng ML"""
        try:
            # Fix division by zero and negative values
            if current_orders <= 0:
                current_orders = 1  # Set minimum value to avoid issues
            
            # Mô phỏng ML model đơn giản
            base_demand = current_orders * trend_factor
            seasonal_factor = 1 + 0.2 * np.sin(datetime.now().month * np.pi / 6)
            predicted_demand = base_demand * seasonal_factor
            return max(1, int(predicted_demand))  # Ensure positive value
        except Exception as e:
            st.error(f"Lỗi dự báo nhu cầu: {str(e)}")
            return max(1, current_orders)  # Fallback value
    
    def generate_solutions(self, problem_type, problem_data):
        """Tạo danh sách giải pháp cho vấn đề"""
        solutions = []
        
        if problem_type == 'INVENTORY_SHORTAGE':
            item = problem_data['item']
            shortage = problem_data['shortage']
            
            solutions = [
                {
                    'id': 'sol_1',
                    'title': f'Nhập khẩu khẩn cấp {item}',
                    'description': f'Nhập ngay {int(shortage * 1.5)} {item} từ nhà cung cấp chính',
                    'benefit_score': 95,
                    'cost': 'Cao',
                    'time': '2-3 ngày',
                    'risk': 'Thấp',
                    'priority': 'best'
                },
                {
                    'id': 'sol_2',
                    'title': f'Tìm nhà cung cấp thay thế',
                    'description': f'Tìm nhà cung cấp mới cho {item} với giá tốt hơn',
                    'benefit_score': 85,
                    'cost': 'Trung bình',
                    'time': '5-7 ngày',
                    'risk': 'Trung bình',
                    'priority': 'medium'
                },
                {
                    'id': 'sol_3',
                    'title': 'Tối ưu hóa sản xuất',
                    'description': f'Giảm sản xuất tạm thời để tiết kiệm {item}',
                    'benefit_score': 70,
                    'cost': 'Thấp',
                    'time': '1-2 ngày',
                    'risk': 'Cao',
                    'priority': 'urgent'
                },
                {
                    'id': 'sol_4',
                    'title': 'Sử dụng hàng tồn kho dự phòng',
                    'description': f'Kích hoạt kho dự phòng cho {item}',
                    'benefit_score': 80,
                    'cost': 'Trung bình',
                    'time': '1 ngày',
                    'risk': 'Thấp',
                    'priority': 'medium'
                }
            ]
        
        elif problem_type == 'ROUTE_RISK':
            route = problem_data['route']
            risk = problem_data['risk']
            
            solutions = [
                {
                    'id': 'sol_1',
                    'title': f'Chuyển sang tuyến thay thế',
                    'description': f'Chuyển từ {route} sang tuyến an toàn hơn',
                    'benefit_score': 90,
                    'cost': 'Trung bình',
                    'time': '1-2 ngày',
                    'risk': 'Thấp',
                    'priority': 'best'
                },
                {
                    'id': 'sol_2',
                    'title': 'Tăng cường bảo hiểm vận chuyển',
                    'description': f'Mua bảo hiểm bổ sung cho {route}',
                    'benefit_score': 75,
                    'cost': 'Cao',
                    'time': '3-5 ngày',
                    'risk': 'Thấp',
                    'priority': 'medium'
                },
                {
                    'id': 'sol_3',
                    'title': 'Đàm phán với đối tác vận chuyển',
                    'description': f'Yêu cầu cải thiện dịch vụ cho {route}',
                    'benefit_score': 65,
                    'cost': 'Thấp',
                    'time': '7-10 ngày',
                    'risk': 'Cao',
                    'priority': 'urgent'
                },
                {
                    'id': 'sol_4',
                    'title': 'Phân tán rủi ro',
                    'description': f'Chia nhỏ lô hàng qua nhiều tuyến',
                    'benefit_score': 85,
                    'cost': 'Trung bình',
                    'time': '2-3 ngày',
                    'risk': 'Trung bình',
                    'priority': 'medium'
                }
            ]
        
        elif problem_type == 'DEMAND_SURGE':
            current = problem_data['current_orders']
            predicted = problem_data['predicted_demand']
            
            solutions = [
                {
                    'id': 'sol_1',
                    'title': 'Tăng cường sản xuất',
                    'description': f'Tăng sản xuất từ {current} lên {predicted} đơn vị',
                    'benefit_score': 95,
                    'cost': 'Cao',
                    'time': '1-2 tuần',
                    'risk': 'Thấp',
                    'priority': 'best'
                },
                {
                    'id': 'sol_2',
                    'title': 'Thuê ngoài sản xuất',
                    'description': f'Ký hợp đồng với đối tác sản xuất bên ngoài',
                    'benefit_score': 80,
                    'cost': 'Cao',
                    'time': '2-3 tuần',
                    'risk': 'Trung bình',
                    'priority': 'medium'
                },
                {
                    'id': 'sol_3',
                    'title': 'Tối ưu hóa quy trình',
                    'description': f'Cải thiện hiệu suất sản xuất hiện tại',
                    'benefit_score': 70,
                    'cost': 'Trung bình',
                    'time': '1 tuần',
                    'risk': 'Thấp',
                    'priority': 'medium'
                },
                {
                    'id': 'sol_4',
                    'title': 'Tăng ca làm việc',
                    'description': f'Yêu cầu nhân viên làm thêm giờ',
                    'benefit_score': 60,
                    'cost': 'Trung bình',
                    'time': '3-5 ngày',
                    'risk': 'Cao',
                    'priority': 'urgent'
                }
            ]
        
        # Sắp xếp theo benefit score (cao nhất lên đầu)
        solutions.sort(key=lambda x: x['benefit_score'], reverse=True)
        return solutions
    
    def create_notification(self, problem_type, problem_data, severity='HIGH'):
        """Tạo notification cho vấn đề được phát hiện"""
        notification_id = self.generate_notification_id()
        created_time = datetime.now()
        
        # Tạo tiêu đề và mô tả
        if problem_type == 'INVENTORY_SHORTAGE':
            title = f"🚨 Thiếu hụt tồn kho: {problem_data['item']}"
            description = f"Tồn kho {problem_data['item']} chỉ còn {problem_data['ratio']:.1%} so với yêu cầu"
        elif problem_type == 'ROUTE_RISK':
            title = f"⚠️ Rủi ro vận chuyển: {problem_data['route']}"
            description = f"Tuyến {problem_data['route']} có rủi ro {problem_data['risk']}%"
        elif problem_type == 'DEMAND_SURGE':
            title = f"📈 Tăng đột biến nhu cầu"
            description = f"Dự báo nhu cầu tăng {((problem_data['predicted_demand'] - problem_data['current_orders']) / problem_data['current_orders']) * 100:.1f}%"
        
        # Tạo danh sách giải pháp
        solutions = self.generate_solutions(problem_type, problem_data)
        
        notification = {
            'id': notification_id,
            'type': problem_type,
            'title': title,
            'description': description,
            'severity': severity,
            'created_time': created_time,
            'deadline': created_time + timedelta(seconds=50),  # 50 giây timeout
            'solutions': solutions,
            'status': 'PENDING',  # PENDING, RESPONDED, AUTO_RESOLVED
            'selected_solution': None,
            'auto_resolved': False
        }
        
        return notification
    
    def detect_problems(self, inventory_data, route_data, current_orders):
        """Phát hiện các vấn đề và tạo notifications"""
        notifications = []
        
        try:
            # 1. Phát hiện thiếu hụt tồn kho
            inventory_risks = self.analyze_inventory_risk(inventory_data)
            for risk in inventory_risks:
                if risk['risk'] == 'HIGH':
                    notification = self.create_notification('INVENTORY_SHORTAGE', risk, 'HIGH')
                    notifications.append(notification)
            
            # 2. Phát hiện rủi ro tuyến vận chuyển
            route_risks = self.analyze_route_risk(route_data)
            for risk in route_risks:
                notification = self.create_notification('ROUTE_RISK', risk, 'MEDIUM')
                notifications.append(notification)
            
            # 3. Phát hiện tăng đột biến nhu cầu
            predicted_demand = self.predict_demand(current_orders)
            if predicted_demand > current_orders * 1.2:
                demand_data = {
                    'current_orders': current_orders,
                    'predicted_demand': predicted_demand
                }
                notification = self.create_notification('DEMAND_SURGE', demand_data, 'MEDIUM')
                notifications.append(notification)
        
        except Exception as e:
            st.error(f"Lỗi phát hiện vấn đề: {str(e)}")
        
        return notifications
    
    def auto_resolve_notification(self, notification):
        """Tự động giải quyết notification sau 50 giây"""
        try:
            # Check if there are any solutions
            if not notification['solutions'] or len(notification['solutions']) == 0:
                notification['result'] = "❌ Không có giải pháp khả dụng để tự động giải quyết."
                notification['auto_resolved'] = True
                notification['status'] = 'AUTO_RESOLVED'
                return notification['result']

            # Chọn giải pháp tốt nhất (đầu tiên trong danh sách đã sắp xếp)
            best_solution = notification['solutions'][0]
            notification['selected_solution'] = best_solution
            notification['status'] = 'AUTO_RESOLVED'
            notification['auto_resolved'] = True

            # Thực thi giải pháp
            result = self.execute_solution(best_solution, notification['type'], notification)

            return result

        except Exception as e:
            return f"❌ Lỗi tự động giải quyết: {str(e)}"
    
    def execute_solution(self, solution, problem_type, notification):
        """Thực thi giải pháp được chọn"""
        try:
            if problem_type == 'INVENTORY_SHORTAGE':
                item = notification['description'].split(': ')[1].split()[0]
                if solution['id'] == 'sol_1':  # Nhập khẩu khẩn cấp
                    # Tìm item trong inventory_data
                    found = False
                    for inv_item, data in st.session_state.inventory_data.items():
                        if item in inv_item:
                            shortage = data['required'] - data['current']
                            st.session_state.inventory_data[inv_item]['current'] += int(shortage * 1.5)
                            found = True
                            return f"✅ Đã nhập khẩn cấp {int(shortage * 1.5)} {inv_item}"
                    if not found:
                        return f"❌ Không tìm thấy linh kiện {item} trong kho."

            elif problem_type == 'ROUTE_RISK':
                route = notification['description'].split(': ')[1]
                if solution['id'] == 'sol_1':  # Chuyển tuyến
                    # Tìm tuyến thay thế tốt nhất
                    if not st.session_state.route_data:
                        return "❌ Không có dữ liệu tuyến vận chuyển."
                    best_route = min(st.session_state.route_data.items(), key=lambda x: x[1]['risk'])
                    if route not in st.session_state.route_data:
                        return f"❌ Không tìm thấy tuyến {route} trong dữ liệu."
                    st.session_state.route_data[route]['risk'] = max(30, st.session_state.route_data[route]['risk'] - 40)
                    st.session_state.route_data[best_route[0]]['risk'] = min(60, best_route[1]['risk'] + 20)
                    return f"✅ Đã chuyển từ {route} sang {best_route[0]}"

            elif problem_type == 'DEMAND_SURGE':
                if solution['id'] == 'sol_1':  # Tăng sản xuất
                    additional_orders = 50
                    st.session_state.orders_processed += additional_orders
                    return f"✅ Đã tăng sản xuất thêm {additional_orders} đơn hàng"

            return f"✅ Đã thực thi giải pháp: {solution['title']}"
        
        except Exception as e:
            return f"❌ Lỗi thực thi giải pháp: {str(e)}"

# Khởi tạo AI Engine
ai_engine = AIDecisionEngine()

# Header
st.markdown('<h1 class="main-header">🤖 AI Dashboard - Quản lý Chuỗi Cung Ứng ABC</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1.1rem;">Sản xuất xe đạp điện thông minh với AI</p>', unsafe_allow_html=True)

# Sidebar điều khiển
st.sidebar.header("🎛️ Điều khiển mô phỏng")

# Nút điều khiển
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("▶️ Bắt đầu", key="start"):
        st.session_state.simulation_running = True
with col2:
    if st.button("⏸️ Dừng", key="pause"):
        st.session_state.simulation_running = False
with col3:
    if st.button("🔄 Reset", key="reset"):
        st.session_state.simulation_running = False
        st.session_state.simulation_step = 0
        st.session_state.orders_processed = 0
        st.session_state.ai_decisions = []
        st.session_state.active_notifications = []
        st.session_state.notification_history = []
        # Reset data
        st.session_state.inventory_data = {
            'Pin lithium': {'current': 2500, 'required': 3500, 'trend': -50},
            'Motor điện': {'current': 1800, 'required': 2200, 'trend': -30},
            'Khung xe': {'current': 3200, 'required': 2800, 'trend': -40},
            'Bánh xe': {'current': 4500, 'required': 4200, 'trend': -60},
            'Phanh đĩa': {'current': 2100, 'required': 2800, 'trend': -35}
        }
        st.session_state.route_data = {
            'Tuyến A (Trung Quốc)': {'risk': 85, 'delay': '7-10 ngày', 'position': 0, 'speed': 2},
            'Tuyến B (Đài Loan)': {'risk': 25, 'delay': '0-2 ngày', 'position': 0, 'speed': 3},
            'Tuyến C (Nhật Bản)': {'risk': 45, 'delay': '3-5 ngày', 'position': 0, 'speed': 2.5}
        }

# Cài đặt AI
st.sidebar.header("🤖 Cài đặt AI")
ai_auto_mode = st.sidebar.checkbox("🔄 Tự động ra quyết định", value=True)
ai_sensitivity = st.sidebar.slider("🎯 Độ nhạy AI", 0.1, 1.0, 0.7, 0.1)

# Auto-refresh settings
st.sidebar.header("⚙️ Cài đặt làm mới")
auto_refresh = st.sidebar.checkbox("🔄 Tự động làm mới", value=True)
refresh_interval = st.sidebar.slider("⏱️ Khoảng thời gian (giây)", 1, 10, 2)

# Debug controls
st.sidebar.header("🐛 Debug Controls")
if st.sidebar.button("🔍 Test Notifications"):
    test_notifications = ai_engine.detect_problems(
        st.session_state.inventory_data,
        st.session_state.route_data,
        st.session_state.orders_processed
    )
    for notification in test_notifications:
        st.session_state.active_notifications.append(notification)
    st.success("✅ Test notifications added!")

# Add Ollama model selection in sidebar
st.sidebar.header("🧠 AI Model Settings")
ollama_model = st.sidebar.text_input("Ollama Model Name", value="llama2")

# Mô phỏng thời gian thực
if st.session_state.simulation_running:
    try:
        # Cập nhật dữ liệu
        st.session_state.simulation_step += 1
        
        # Cập nhật tồn kho (giảm theo thời gian)
        for item in st.session_state.inventory_data:
            decrease = random.randint(10, 30)
            st.session_state.inventory_data[item]['current'] = max(0, 
                st.session_state.inventory_data[item]['current'] - decrease)
        
        # Cập nhật vị trí xe
        for route in st.session_state.route_data:
            st.session_state.route_data[route]['position'] += st.session_state.route_data[route]['speed']
            if st.session_state.route_data[route]['position'] >= 100:
                st.session_state.route_data[route]['position'] = 0
        
        # Cập nhật đơn hàng
        st.session_state.orders_processed += random.randint(1, 5)
        
        # AI phát hiện vấn đề và tạo notifications
        if st.session_state.simulation_step % 15 == 0:  # Mỗi 15 bước
            new_notifications = ai_engine.detect_problems(
                st.session_state.inventory_data,
                st.session_state.route_data,
                st.session_state.orders_processed
            )
            
            for notification in new_notifications:
                # Kiểm tra xem notification đã tồn tại chưa
                existing_ids = [n['id'] for n in st.session_state.active_notifications]
                if notification['id'] not in existing_ids:
                    st.session_state.active_notifications.append(notification)
        
        # Kiểm tra timeout cho notifications
        current_time = datetime.now()
        for notification in st.session_state.active_notifications[:]:  # Copy list để tránh lỗi khi modify
            if notification['status'] == 'PENDING' and current_time >= notification['deadline']:
                # Auto resolve sau 45 phút
                result = ai_engine.auto_resolve_notification(notification)
                notification['result'] = result
                st.session_state.notification_history.append(notification)
                st.session_state.active_notifications.remove(notification)
    
    except Exception as e:
        st.error(f"Lỗi trong mô phỏng: {str(e)}")
        st.session_state.simulation_running = False

# Hiển thị trạng thái mô phỏng
if st.session_state.simulation_running:
    st.markdown(f"""
    <div class="alert-box">
        <strong>🔄 Đang mô phỏng...</strong> Bước {st.session_state.simulation_step} | 
        Đơn hàng đã xử lý: {st.session_state.orders_processed}
    </div>
    """, unsafe_allow_html=True)

# Debug Panel
with st.expander("🐛 Debug Information"):
    st.write(f"**Simulation Step:** {st.session_state.simulation_step}")
    st.write(f"**Orders Processed:** {st.session_state.orders_processed}")
    st.write(f"**Active Notifications:** {len(st.session_state.active_notifications)}")
    st.write(f"**Notification History:** {len(st.session_state.notification_history)}")
    
    if st.session_state.active_notifications:
        st.write("**Active Notification IDs:**")
        for notification in st.session_state.active_notifications:
            st.write(f"- {notification['id']}: {notification['title']}")
    
    st.write("**Current Inventory:**")
    for item, data in st.session_state.inventory_data.items():
        ratio = data['current'] / data['required'] if data['required'] > 0 else 0
        st.write(f"- {item}: {data['current']}/{data['required']} ({ratio:.1%})")

# AI Notifications Panel
st.subheader("🔔 Thông báo AI - Cần xử lý")

# Helper function to interact with Ollama
def ollama_chat(messages, model_name):
    try:
        response = ollama.chat(
            model=model_name,
            messages=messages
        )
        return response['message']['content']
    except Exception as e:
        return f"[Lỗi Ollama]: {str(e)}"

# Hiển thị active notifications
if st.session_state.active_notifications:
    for notification in st.session_state.active_notifications:
        if notification['status'] == 'PENDING':
            notif_id = notification['id']
            # Initialize chat history for this notification
            if notif_id not in st.session_state.notification_chats:
                # System prompt with context
                system_prompt = (
                    f"Bạn là AI chuyên gia chuỗi cung ứng. Dưới đây là một vấn đề cần giải quyết: {notification['title']} - {notification['description']}. "
                    f"Các giải pháp khả dụng: " + ", ".join([f'{i+1}. {s['title']}: {s['description']}' for i, s in enumerate(notification['solutions'])]) + ". "
                    "Hãy thảo luận với người dùng để hiểu rõ hơn về tình huống, sau đó tự động chọn giải pháp tốt nhất và thông báo quyết định của bạn. Khi đã quyết định, hãy bắt đầu câu trả lời bằng [DECISION] và ghi rõ giải pháp bạn chọn."
                )
                st.session_state.notification_chats[notif_id] = [
                    {"role": "system", "content": system_prompt}
                ]
            chat_history = st.session_state.notification_chats[notif_id]

            # Display notification info
            time_left = notification['deadline'] - datetime.now()
            seconds_left = max(0, int(time_left.total_seconds()))
            st.markdown(f"""
            <div class="notification-box">
                <h4>{notification['title']}</h4>
                <p>{notification['description']}</p>
                <div class="timer-box">
                    ⏰ Còn lại: {seconds_left} giây
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display chat history
            st.write("**💬 AI Chat:**")
            for msg in chat_history[1:]:  # skip system prompt
                if msg['role'] == 'user':
                    st.markdown(f"<div style='color:blue'><b>Bạn:</b> {msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='color:green'><b>AI:</b> {msg['content']}</div>", unsafe_allow_html=True)

            # User input
            user_input = st.text_input(f"Nhập câu hỏi hoặc ý kiến cho AI (ID: {notif_id})", key=f"chat_input_{notif_id}")
            send_btn = st.button("Gửi", key=f"send_btn_{notif_id}")

            if send_btn and user_input.strip():
                chat_history.append({"role": "user", "content": user_input.strip()})
                with st.spinner("AI đang trả lời..."):
                    ai_response = ollama_chat(chat_history, ollama_model)
                chat_history.append({"role": "assistant", "content": ai_response})
                st.session_state.notification_chats[notif_id] = chat_history
                st.rerun()

            # Check if AI made a decision
            if any(msg['role'] == 'assistant' and msg['content'].strip().startswith('[DECISION]') for msg in chat_history):
                # Find the latest decision message
                decision_msg = next(msg for msg in reversed(chat_history) if msg['role'] == 'assistant' and msg['content'].strip().startswith('[DECISION]'))
                # Try to match the solution from the message
                chosen_solution = None
                for sol in notification['solutions']:
                    if sol['title'] in decision_msg['content']:
                        chosen_solution = sol
                        break
                if not chosen_solution:
                    # fallback: pick the first solution
                    chosen_solution = notification['solutions'][0]
                notification['selected_solution'] = chosen_solution
                notification['status'] = 'RESPONDED'
                result = ai_engine.execute_solution(chosen_solution, notification['type'], notification)
                notification['result'] = result
                st.session_state.notification_history.append(notification)
                st.session_state.active_notifications.remove(notification)
                st.success(f"🤖 AI đã chọn giải pháp: {chosen_solution['title']}")
                st.rerun()

            st.write("---")

# Lịch sử notifications
if st.session_state.notification_history:
    st.subheader("📋 Lịch sử thông báo")
    for notification in st.session_state.notification_history[-5:]:  # Hiển thị 5 cái gần nhất
        status_icon = "🤖" if notification['auto_resolved'] else "👤"
        status_text = "Tự động giải quyết" if notification['auto_resolved'] else "Xử lý thủ công"
        
        st.write(f"**{status_icon} {notification['title']}** - {status_text}")
        st.write(f"Thời gian: {notification['created_time'].strftime('%H:%M:%S')}")
        if notification['selected_solution']:
            st.write(f"Giải pháp: {notification['selected_solution']['title']}")
        if 'result' in notification:
            st.write(f"Kết quả: {notification['result']}")
        st.write("---")

# KPI Cards
st.subheader("📊 Chỉ số hiệu suất chính")
col1, col2, col3, col4 = st.columns(4)

with col1:
    demand_increase = 40
    st.metric("📈 Dự báo nhu cầu", f"+{demand_increase}%", delta=f"+{demand_increase}%")

with col2:
    shortage_items = sum(1 for item, data in st.session_state.inventory_data.items() 
                        if data['current'] < data['required'])
    st.metric("📦 Linh kiện thiếu", f"{shortage_items}/5", 
              delta=f"-{5-shortage_items}" if shortage_items > 0 else "0")

with col3:
    high_risk_routes = sum(1 for route, data in st.session_state.route_data.items() 
                          if data['risk'] >= 70)
    st.metric("🚛 Tuyến rủi ro cao", f"{high_risk_routes}/3", 
              delta=f"-{3-high_risk_routes}" if high_risk_routes > 0 else "0")

with col4:
    on_time_delivery = 87
    st.metric("⏰ Giao đúng hẹn", f"{on_time_delivery}%", delta=f"+{on_time_delivery-80}%")

# Layout chính
col1, col2 = st.columns([2, 1])

with col1:
    # Biểu đồ dự báo nhu cầu
    st.subheader("📈 Dự báo nhu cầu AI (ML Model)")
    
    # Tạo dữ liệu mẫu
    months = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    actual = [1200, 1350, 1400, None, None, None]
    predicted = [1150, 1280, 1420, 1580, 1650, 1720]
    optimized = [1680, 1792, 1960, 2212, 2310, 2408]
    
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(x=months, y=actual, name='Thực tế', line=dict(color='gray')))
    fig_demand.add_trace(go.Scatter(x=months, y=predicted, name='ML Dự báo', line=dict(color='blue')))
    fig_demand.add_trace(go.Scatter(x=months, y=optimized, name='AI Tối ưu (+40%)', line=dict(color='green', width=3)))
    
    fig_demand.update_layout(
        title="Dự báo nhu cầu xe đạp điện",
        xaxis_title="Tháng",
        yaxis_title="Số lượng",
        height=400
    )
    st.plotly_chart(fig_demand, use_container_width=True)

with col2:
    # Trạng thái tồn kho
    st.subheader("📦 Tình trạng tồn kho")
    
    for item, data in st.session_state.inventory_data.items():
        # Fix division by zero error
        if data['required'] > 0:
            ratio = data['current'] / data['required']
        else:
            ratio = 0
            
        color = 'red' if ratio < 0.7 else 'orange' if ratio < 0.9 else 'green'
        status = 'Thiếu' if ratio < 0.7 else 'Ít' if ratio < 0.9 else 'Đủ'
        
        st.write(f"**{item}**")
        st.progress(min(ratio, 1.0))
        st.write(f"Hiện tại: {data['current']:,} | Cần: {data['required']:,} | Trạng thái: {status}")
        st.write("---")

# Phân tích rủi ro vận chuyển
st.subheader("🚛 Phân tích rủi ro vận chuyển (Deep Learning)")

for route, data in st.session_state.route_data.items():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.write(f"**{route}**")
        risk_color = 'red' if data['risk'] >= 70 else 'orange' if data['risk'] >= 40 else 'green'
        st.write(f"Rủi ro: {data['risk']:.0f}% | Dự kiến trễ: {data['delay']}")
        
        # Thanh tiến trình xe
        progress_bar = st.progress(data['position'] / 100)
        st.write(f"🚛 Tiến trình: {data['position']:.1f}%")
    
    with col2:
        st.metric("Rủi ro", f"{data['risk']:.0f}%")
    
    with col3:
        st.metric("Vị trí", f"{data['position']:.1f}%")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>🤖 <strong>AI Supply Chain Management System</strong> | Powered by Machine Learning & Deep Learning</p>
    <p>Công ty ABC - Sản xuất xe đạp điện thông minh</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh mechanism (only if simulation is running and auto-refresh is enabled)
if st.session_state.simulation_running and auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()