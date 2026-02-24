#!/usr/bin/env python3
"""
依赖检查脚本 - 检查 AlphaGPT 项目的 Python 依赖更新情况
"""

import subprocess
import sys
from datetime import datetime
import json

def run_command(cmd):
    """执行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timeout", -1
    except Exception as e:
        return "", str(e), -1

def check_pip_outdated():
    """检查过时的依赖包"""
    print("=" * 60)
    print("📦 检查过时的 Python 依赖包")
    print("=" * 60)

    stdout, stderr, code = run_command("pip list --outdated --format=json")

    if code != 0:
        print(f"❌ 执行失败: {stderr}")
        return []

    try:
        outdated = json.loads(stdout)
        if not outdated:
            print("✅ 所有依赖包都是最新版本！")
            return []

        print(f"\n📋 发现 {len(outdated)} 个可更新的包:\n")
        for pkg in outdated:
            print(f"  • {pkg['name']:<30} {pkg['version']:<12} -> {pkg['latest_version']:<12}")

        return outdated
    except json.JSONDecodeError:
        print("❌ 无法解析输出")
        return []

def check_security():
    """检查安全漏洞"""
    print("\n" + "=" * 60)
    print("🔒 检查安全漏洞")
    print("=" * 60)

    stdout, stderr, code = run_command("pip-audit --desc --format=json 2>/dev/null")

    if code == 127:
        print("⚠️  pip-audit 未安装，跳过安全检查")
        print("   安装命令: pip install pip-audit")
        return []
    elif code != 0:
        print(f"❌ 执行失败: {stderr}")
        return []

    try:
        vulnerabilities = json.loads(stdout)
        if not vulnerabilities:
            print("✅ 未发现已知安全漏洞！")
            return []

        print(f"\n🚨 发现 {len(vulnerabilities)} 个安全漏洞:\n")
        for vuln in vulnerabilities[:10]:  # 只显示前 10 个
            print(f"  • {vuln['name']:<30} {vuln['affected_versions']}")
            print(f"    修复版本: {vuln['fix_versions']}")
            print(f"    详情: {vuln['advisory']}")
            print()

        return vulnerabilities
    except json.JSONDecodeError:
        print("❌ 无法解析输出")
        return []

def main():
    """主函数"""
    print(f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔍 AlphaGPT 依赖检查")
    print("=" * 60)

    # 检查过时的依赖
    outdated = check_pip_outdated()

    # 检查安全漏洞
    vulnerabilities = check_security()

    # 生成总结
    print("\n" + "=" * 60)
    print("📊 检查总结")
    print("=" * 60)

    total_outdated = len(outdated)
    total_vulns = len(vulnerabilities)

    if total_outdated == 0 and total_vulns == 0:
        print("✅ 所有依赖包都是最新版本，没有已知安全漏洞！")
        return 0
    else:
        if total_outdated > 0:
            print(f"⚠️  {total_outdated} 个包可更新")
        if total_vulns > 0:
            print(f"🚨 {total_vulns} 个安全漏洞")

        print("\n💡 建议:")
        if total_vulns > 0:
            print("   1. 优先修复安全漏洞")
            if total_outdated > 0:
                print("   2. 评估并更新过时的依赖包")
        elif total_outdated > 0:
            print("   1. 查看过时的依赖包详情")
            print("   2. 评估升级的必要性和风险")

        return 1

if __name__ == "__main__":
    sys.exit(main())
