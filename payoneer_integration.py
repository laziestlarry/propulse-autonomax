# payoneer_integration.py
import payoneer_sdk

def auto_withdraw(currency, amount):
    account = payoneer_sdk.Account(currency=currency)
    if account.balance >= amount:
        account.withdraw_to_bank(
            amount=amount,
            conversion_rule="optimize_for_low_fees"
        )
    return f"Withdrew {amount} {currency} to Payoneer"