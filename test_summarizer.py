from src.summerizer import generate_summary

result = generate_summary("AAPL")
print(result)  # Print full dict first to see what went wrong

if "summary" in result:
    print("\nSummary:\n", result["summary"])
else:
    print("\n❌ Summary generation failed.")
