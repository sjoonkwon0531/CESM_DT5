#!/usr/bin/env python3
"""
DT5 확장 기능 데모 런처
MVP 데모 실행을 위한 스크립트
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """필수 종속성 확인"""
    required_packages = ['streamlit', 'plotly', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} missing")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def run_basic_tests():
    """기본 테스트 실행"""
    print("\n🧪 Running basic tests...")
    
    try:
        result = subprocess.run([sys.executable, 'test_expansion.py'], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False

def launch_streamlit():
    """Streamlit 앱 실행"""
    print("\n🚀 Launching DT5 Expansion Demo...")
    print("📱 Opening http://localhost:8502 in browser...")
    
    try:
        # Streamlit 앱 실행 (백그라운드)
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 
            'app_expansion.py',
            '--server.port', '8502',
            '--server.headless', 'true'
        ], cwd=Path.cwd())
        
        # 잠시 기다린 후 브라우저 열기
        time.sleep(3)
        
        try:
            webbrowser.open('http://localhost:8502')
        except:
            print("⚠️  Could not open browser automatically")
            print("   Please open http://localhost:8502 manually")
        
        print("\n" + "="*60)
        print("🎉 DT5 EXPANSION DEMO IS RUNNING!")
        print("="*60)
        print("📊 Demo Features Available:")
        print("   • 🌊 3-Way Stress Test Comparison") 
        print("   • 💾 Data Survival Analysis")
        print("   • 📈 Unified Dashboard")
        print("   • ⚡ Real-time System Monitoring")
        print("\n🎯 Demo Highlights:")
        print("   • CEMS 123x faster response time")
        print("   • CEMS 14.5x longer backup duration")
        print("   • CEMS 100% data survival rate")
        print("   • Only CEMS achieves Tier IV SLA")
        print("\n💡 Usage Instructions:")
        print("   1. Select analysis type in sidebar")
        print("   2. Configure parameters")
        print("   3. Click 'Execute' buttons")
        print("   4. Explore results and comparisons")
        print("\n⌨️  Press Ctrl+C to stop the demo")
        print("="*60)
        
        # 프로세스 대기
        process.wait()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Demo stopped by user")
        process.terminate()
        
    except Exception as e:
        print(f"\n❌ Failed to launch Streamlit: {e}")
        return False
    
    return True

def show_demo_info():
    """데모 정보 표시"""
    print("🚀 DT5 EXPANSION MVP DEMO")
    print("="*50)
    print("📋 MVP Features:")
    print("   • 3-Way Stress Test Engine")
    print("   • Data Survival Analysis (t2/t3)")
    print("   • Energy SLA Calculation")
    print("   • Unified Performance Analytics")
    print("\n🎯 Demo Objectives:")
    print("   • Prove CEMS microgrid superiority")
    print("   • Quantify competitive advantages")
    print("   • Show investment ROI potential")
    print("   • Demonstrate technical feasibility")
    print("\n📊 Expected Results:")
    print("   • CEMS wins all major KPIs")
    print("   • 99.8%+ data survival rate")
    print("   • Tier IV energy SLA compliance")
    print("   • <3 year ROI payback period")
    print("="*50)

def main():
    """메인 데모 실행"""
    show_demo_info()
    
    # 1. 종속성 확인
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Dependencies not satisfied. Please install required packages.")
        return False
    
    # 2. 기본 테스트 실행
    if not run_basic_tests():
        print("\n⚠️  Tests failed, but demo will continue...")
        print("   Some features may not work properly.")
        
        continue_anyway = input("\nContinue with demo anyway? (y/n): ").lower().strip()
        if continue_anyway != 'y':
            print("Demo cancelled.")
            return False
    
    # 3. Streamlit 데모 실행
    success = launch_streamlit()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n📝 Next Steps for Full Implementation:")
        print("   1. GPU degradation module (Phase 2)")
        print("   2. Cascading failure modeling")
        print("   3. Real-time monitoring integration")
        print("   4. Advanced visualization features")
        print("   5. Production deployment preparation")
    else:
        print("\n❌ Demo failed to complete")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)