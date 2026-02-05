# app.py
from verify import verify_user


def main():
    print("\n==============================")
    print(" KYC FACE VERIFICATION DEMO ")
    print("==============================")

    while True:
        print("\nOptions:")
        print("1 → Verify Identity")
        print("2 → Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            verify_user()
        elif choice == "2":
            print("👋 Exiting demo")
            break
        else:
            print("❌ Invalid choice")


if __name__ == "__main__":
    main()
