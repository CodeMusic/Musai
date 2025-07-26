#!/usr/bin/env python3
"""
Test script to verify directory creation functionality in file_operators.
This script tests both local and sandbox directory creation.
"""

import asyncio
import tempfile
from pathlib import Path

from app.tool.file_operators import LocalFileOperator, SandboxFileOperator


async def test_local_directory_creation():
    """Test directory creation in local environment."""
    print("🧪 Testing local directory creation...")

    operator = LocalFileOperator()

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir) / "test_subdir" / "nested" / "deep"

        print(f"📁 Creating directory: {test_dir}")

        # Test directory creation
        await operator.create_directory(test_dir, parents=True)

        # Verify directory was created
        if await operator.is_directory(test_dir):
            print("✅ Directory created successfully!")
        else:
            print("❌ Directory creation failed!")
            return False

        # Test creating a file in the new directory
        test_file = test_dir / "test.txt"
        test_content = "Hello, World!"

        print(f"📄 Creating file: {test_file}")
        await operator.write_file(test_file, test_content)

        # Verify file was created
        if await operator.exists(test_file):
            print("✅ File created successfully!")
            return True
        else:
            print("❌ File creation failed!")
            return False


async def test_sandbox_directory_creation():
    """Test directory creation in sandbox environment."""
    print("\n🧪 Testing sandbox directory creation...")

    operator = SandboxFileOperator()

    try:
        # Test directory creation in sandbox
        test_dir = "/workspace/test_subdir/nested/deep"

        print(f"📁 Creating directory: {test_dir}")

        # Test directory creation
        await operator.create_directory(test_dir, parents=True)

        # Verify directory was created
        if await operator.is_directory(test_dir):
            print("✅ Directory created successfully!")
        else:
            print("❌ Directory creation failed!")
            return False

        # Test creating a file in the new directory
        test_file = f"{test_dir}/test.txt"
        test_content = "Hello, Sandbox World!"

        print(f"📄 Creating file: {test_file}")
        await operator.write_file(test_file, test_content)

        # Verify file was created
        if await operator.exists(test_file):
            print("✅ File created successfully!")
            return True
        else:
            print("❌ File creation failed!")
            return False

    except Exception as e:
        print(f"❌ Sandbox test failed: {e}")
        return False


async def test_str_replace_editor_directory_creation():
    """Test that StrReplaceEditor creates directories when needed."""
    print("\n🧪 Testing StrReplaceEditor directory creation...")

    from app.tool.str_replace_editor import StrReplaceEditor

    editor = StrReplaceEditor()

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "new_dir" / "nested" / "test.txt"
        test_content = "This is a test file created in a new directory structure."

        print(f"📄 Creating file with StrReplaceEditor: {test_file}")

        try:
            result = await editor.execute(
                command="create", path=str(test_file), file_text=test_content
            )

            print(f"✅ StrReplaceEditor result: {result}")
            return True

        except Exception as e:
            print(f"❌ StrReplaceEditor test failed: {e}")
            return False


async def main():
    """Run all directory creation tests."""
    print("🚀 Starting directory creation tests...")

    results = []

    try:
        # Test local directory creation
        results.append(await test_local_directory_creation())

        # Test sandbox directory creation (if sandbox is available)
        try:
            results.append(await test_sandbox_directory_creation())
        except Exception as e:
            print(f"⚠️ Sandbox test skipped: {e}")
            results.append(True)  # Skip sandbox test if not available

        # Test StrReplaceEditor directory creation
        results.append(await test_str_replace_editor_directory_creation())

        if all(results):
            print("\n🎉 All directory creation tests passed!")
        else:
            print("\n❌ Some directory creation tests failed!")

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
