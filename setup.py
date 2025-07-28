#!/usr/bin/env python3
"""
Setup script for the Semantic Search Pipeline
Helps users check their environment and start required services.
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def print_header():
    """Print welcome header."""
    print("🔍 " + "="*60)
    print("   Semantic Search Pipeline Setup")
    print("   Using Qwen3-Embedding-0.6B + Qdrant + Gradio")
    print("="*62)
    print()

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.9+)")
        return False

def check_docker():
    """Check if Docker is available."""
    print("🐳 Checking Docker...")
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Docker not found or not working")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ Docker not found")
        return False

def check_poetry():
    """Check if Poetry is available."""
    print("📦 Checking Poetry...")
    try:
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"   ✅ {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Poetry not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ Poetry not found")
        print("   💡 Install with: curl -sSL https://install.python-poetry.org | python3 -")
        return False

def check_env_file():
    """Check if .env file exists."""
    print("⚙️  Checking environment configuration...")
    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content and not content.count('your_openai_api_key_here'):
                print("   ✅ .env file found with OpenAI API key")
                return True
            else:
                print("   ⚠️  .env file found but OpenAI API key not set")
                return False
    else:
        print("   ❌ .env file not found")
        print("   💡 Create .env file with: OPENAI_API_KEY=your_key_here")
        return False

def start_qdrant():
    """Start Qdrant using Docker."""
    print("🚀 Starting Qdrant vector database...")
    try:
        # Check if Qdrant is already running
        try:
            response = requests.get('http://localhost:6333/collections', timeout=3)
            if response.status_code == 200:
                print("   ✅ Qdrant is already running")
                return True
        except requests.RequestException:
            pass
        
        # Start Qdrant
        result = subprocess.run([
            'docker', 'run', '-d', 
            '--name', 'qdrant-semantic-search',
            '-p', '6333:6333',
            'qdrant/qdrant'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   🔄 Starting Qdrant container...")
            # Wait for Qdrant to be ready
            for i in range(30):
                try:
                    response = requests.get('http://localhost:6333/collections', timeout=1)
                    if response.status_code == 200:
                        print("   ✅ Qdrant is ready!")
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
                print(f"   ⏳ Waiting for Qdrant... ({i+1}/30)")
            
            print("   ⚠️  Qdrant started but not responding")
            return False
        else:
            # Container might already exist
            if "already in use" in result.stderr:
                # Try to start existing container
                start_result = subprocess.run(['docker', 'start', 'qdrant-semantic-search'], 
                                            capture_output=True, text=True)
                if start_result.returncode == 0:
                    print("   ✅ Started existing Qdrant container")
                    return True
            
            print(f"   ❌ Failed to start Qdrant: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error starting Qdrant: {e}")
        return False

def install_dependencies():
    """Install Python dependencies using Poetry."""
    print("📚 Installing dependencies...")
    try:
        result = subprocess.run(['poetry', 'install'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("   ✅ Dependencies installed successfully")
            return True
        else:
            print(f"   ❌ Failed to install dependencies: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("   ❌ Installation timed out")
        return False
    except Exception as e:
        print(f"   ❌ Error installing dependencies: {e}")
        return False

def test_system():
    """Test if the system components are working."""
    print("🧪 Testing system components...")
    try:
        # Test import of main modules
        sys.path.insert(0, '.')
        from config import validate_config
        from indexing import verify_indexing_setup
        
        # Validate configuration
        validate_config()
        print("   ✅ Configuration validation passed")
        
        # Verify setup
        setup_status = verify_indexing_setup()
        status = setup_status['overall_status']
        
        if status == 'ready':
            print("   ✅ All components ready")
            return True
        elif status == 'partial':
            print("   ⚠️  System partially ready")
            for issue in setup_status.get('issues', []):
                print(f"   ⚠️  {issue}")
            return True
        else:
            print("   ❌ System not ready")
            for issue in setup_status.get('issues', []):
                print(f"   ❌ {issue}")
            return False
            
    except Exception as e:
        print(f"   ❌ System test failed: {e}")
        return False

def create_env_template():
    """Create a template .env file."""
    env_content = """# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration (optional - defaults are provided)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Example:
# OPENAI_API_KEY=sk-proj-1234567890abcdef...
# QDRANT_HOST=192.168.1.100
# QDRANT_PORT=6333
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("   ✅ Created .env template file")
        print("   💡 Please edit .env and add your OpenAI API key")
        return True
    except Exception as e:
        print(f"   ❌ Failed to create .env file: {e}")
        return False

def main():
    """Main setup function."""
    print_header()
    
    # Check prerequisites
    checks = {
        'Python': check_python_version(),
        'Docker': check_docker(),
        'Poetry': check_poetry(),
        'Environment': check_env_file()
    }
    
    print("\n📋 Prerequisites Summary:")
    for check, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check}")
    
    # Create .env if missing
    if not checks['Environment']:
        print("\n📝 Setting up environment configuration...")
        create_env_template()
    
    # Install dependencies if Poetry is available
    if checks['Poetry']:
        print("\n📦 Installing dependencies...")
        install_dependencies()
    
    # Start Qdrant if Docker is available
    if checks['Docker']:
        print("\n🐳 Setting up Qdrant...")
        start_qdrant()
    
    # Test the system
    print("\n🔧 Testing system...")
    system_ready = test_system()
    
    # Final summary
    print("\n" + "="*62)
    if system_ready:
        print("🎉 Setup completed successfully!")
        print("\n🚀 To start the application:")
        print("   poetry run python main.py")
        print("\n🌐 The interface will be available at:")
        print("   http://localhost:7860")
    else:
        print("⚠️  Setup completed with issues")
        print("\n🔧 Please fix the issues above and run:")
        print("   python setup.py")
        print("\n📖 For more help, check the README.md file")
    
    print("="*62)

if __name__ == "__main__":
    main() 