/*
 * Created by Minseok Chu on 12/29/23.
 */

#include "book_keeper_ui.h"

#include <iostream>
#include <cstdlib>

#ifdef WIN32
#include <optional>
#endif

#include "book_keeper.h"

namespace book_keeper {

BookKeeperUI::BookKeeperUI(book_keeper::BookKeeper* bookKeeper) {
  this->bookKeeper = bookKeeper;
}

void BookKeeperUI::ShowMenu() {
  std::cout << "--- �α� ���α׷� ---" << std::endl;
  std::cout << "1. �ŷ� ����" << std::endl;
  std::cout << "2. �ŷ� �߰�" << std::endl;
  std::cout << "3. �ŷ� ����" << std::endl;
  std::cout << "4. �ŷ� ����" << std::endl;
  std::cout << "5. ����" << std::endl;
  std::cout << "6. ���α׷� ����" << std::endl;
}

void BookKeeperUI::ShowTransaction(
    book_keeper::TransactionInfo& transactionInfo) {
  std::cout << "[" << transactionInfo.id << "] " << transactionInfo.left_entry
            << ", " << transactionInfo.left_entry_comment << " : "
            << transactionInfo.right_entry << ", "
            << transactionInfo.right_entry_comment << " ; "
            << transactionInfo.transaction_comment << std::endl;
}

void BookKeeperUI::ShowTransactions(int row_begin, int row_end) {
  std::optional<book_keeper::TransactionInfo> transactionInfo;

  this->bookKeeper->Sort();

  for (int i = row_begin; i < row_end; ++i) {
    transactionInfo = this->bookKeeper->ReadByIndex(i);

    if (!transactionInfo.has_value()) {
      return;
    } else {
      ShowTransaction(transactionInfo.value());
    }
  }

  system("pause");
}

void BookKeeperUI::AddTransactionFromConsole() {
  book_keeper::TransactionInfo info;

  std::cout << "-- ���� �Է� --" << std::endl;

  std::cout << "���� : ";
  std::cin >> info.left_entry;

  std::cout << "���� : ";
  std::cin >> info.right_entry;

  std::cout << "���� ���� : ";
  std::cin >> info.left_entry_comment;

  std::cout << "���� ���� : ";
  std::cin >> info.right_entry_comment;

  std::cout << "�ŷ� ���� : ";
  std::cin >> info.transaction_comment;

  this->bookKeeper->Create(info);
}

void BookKeeperUI::ModifyTransactionFromConsole() {
  int transaction_id;

  std::cout << "-- ���� ���� --" << std::endl;

  std::cout << "�ŷ� ���̵� : ";
  std::cin >> transaction_id;

  std::optional<book_keeper::TransactionInfo> current_transaction;
  current_transaction = this->bookKeeper->Read(transaction_id);

  if (!current_transaction.has_value()) {
    std::cout << "�ŷ� �����͸� ã�� �� ����, ������ �� �����ϴ�!" << std::endl;
    return;
  }

  std::cout << "���� ������ : " << std::endl;
  this->ShowTransaction(current_transaction.value());

  std::string confirm;

  std::cout << "�����Ͻðڽ��ϱ�? [��/�ƴϿ�]";
  std::cin >> confirm;

  if (!std::equal(confirm.begin(), confirm.end(), "��")) {
    return;
  }

  book_keeper::TransactionInfo info;

  info.id = transaction_id;

  std::cout << "���� : ";
  std::cin >> info.left_entry;

  std::cout << "���� : ";
  std::cin >> info.right_entry;

  std::cout << "���� ���� : ";
  std::cin >> info.left_entry_comment;

  std::cout << "���� ���� : ";
  std::cin >> info.right_entry_comment;

  std::cout << "�ŷ� ���� : ";
  std::cin >> info.transaction_comment;

  this->bookKeeper->Update(info);
}

void BookKeeperUI::DeleteTransactionFromConsole() {
  std::optional<book_keeper::TransactionInfo> transaction_info;
  int transaction_id;

  std::cout << "���� ���� : " << std::endl;

  std::cout << "�ŷ� ���̵�";
  std::cin >> transaction_id;

  transaction_info = this->bookKeeper->Read(transaction_id);

  if (!transaction_info.has_value()) {
    std::cout << "�̹� �����Ǿ��ų� �߸��� �ŷ� �������Դϴ�!" << std::endl;
    return;
  }

  std::cout << "������ �����ʹ� ������ �����ϴ� : " << std::endl;

  this->ShowTransaction(transaction_info.value());

  std::string confirm;

  std::cout << "�����Ͻðڽ��ϱ�? [��/�ƴϿ�]";
  std::cin >> confirm;

  if (!std::equal(confirm.begin(), confirm.end(), "��")) {
    return;
  }

  bookKeeper->Delete(transaction_id);
}

void BookKeeperUI::ClearConsole() {
#ifdef WIN32
  system("cls");
#elif
  system("clear");
#endif
}

void BookKeeperUI::ShowSettlementDetails() {
  std::optional<book_keeper::Settlement> settlement = this->bookKeeper->GetSettlement();

  if (!settlement.has_value()) {
    return;
  }

  std::cout << "���� : " << settlement->total_credit << std::endl;
  std::cout << "�뺯 : " << settlement->total_debit << std::endl;

  system("pause");
}

void BookKeeperUI::StartConsoleApp() {
  int menu_selection;

  while (true) {
    this->ShowMenu();
    std::cout << "�Է� : ";

    std::cin >> menu_selection;

    switch (menu_selection) {
      case 1:
        this->ShowTransactions(0, this->bookKeeper->GetSize());
        this->ClearConsole();
        break;
      case 2:
        this->AddTransactionFromConsole();
        this->ClearConsole();
        break;
      case 3:
        this->ModifyTransactionFromConsole();
        this->ClearConsole();
        break;
      case 4:
        this->DeleteTransactionFromConsole();
        this->ClearConsole();
        break;
      case 5:
        this->ShowSettlementDetails();
        this->ClearConsole();
        break;
      case 6:
        return;
      default:
        this->ClearConsole();
        std::cout << "�߸� �Է��ϼ̽��ϴ�. �ٽ� �õ��ϼ���!" << std::endl;
        std::cin.ignore();
    }
  }
}

}  // namespace book_keeper