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
  std::cout << "--- 부기 프로그램 ---" << std::endl;
  std::cout << "1. 거래 보기" << std::endl;
  std::cout << "2. 거래 추가" << std::endl;
  std::cout << "3. 거래 수정" << std::endl;
  std::cout << "4. 거래 삭제" << std::endl;
  std::cout << "5. 정산" << std::endl;
  std::cout << "6. 프로그램 종료" << std::endl;
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

  std::cout << "-- 정보 입력 --" << std::endl;

  std::cout << "우항 : ";
  std::cin >> info.left_entry;

  std::cout << "좌항 : ";
  std::cin >> info.right_entry;

  std::cout << "좌항 설명 : ";
  std::cin >> info.left_entry_comment;

  std::cout << "우항 설명 : ";
  std::cin >> info.right_entry_comment;

  std::cout << "거래 설명 : ";
  std::cin >> info.transaction_comment;

  this->bookKeeper->Create(info);
}

void BookKeeperUI::ModifyTransactionFromConsole() {
  int transaction_id;

  std::cout << "-- 정보 수정 --" << std::endl;

  std::cout << "거래 아이디 : ";
  std::cin >> transaction_id;

  std::optional<book_keeper::TransactionInfo> current_transaction;
  current_transaction = this->bookKeeper->Read(transaction_id);

  if (!current_transaction.has_value()) {
    std::cout << "거래 데이터를 찾을 수 없어, 수정할 수 없습니다!" << std::endl;
    return;
  }

  std::cout << "기존 데이터 : " << std::endl;
  this->ShowTransaction(current_transaction.value());

  std::string confirm;

  std::cout << "수정하시겠습니까? [예/아니요]";
  std::cin >> confirm;

  if (!std::equal(confirm.begin(), confirm.end(), "예")) {
    return;
  }

  book_keeper::TransactionInfo info;

  info.id = transaction_id;

  std::cout << "우항 : ";
  std::cin >> info.left_entry;

  std::cout << "좌항 : ";
  std::cin >> info.right_entry;

  std::cout << "좌항 설명 : ";
  std::cin >> info.left_entry_comment;

  std::cout << "우항 설명 : ";
  std::cin >> info.right_entry_comment;

  std::cout << "거래 설명 : ";
  std::cin >> info.transaction_comment;

  this->bookKeeper->Update(info);
}

void BookKeeperUI::DeleteTransactionFromConsole() {
  std::optional<book_keeper::TransactionInfo> transaction_info;
  int transaction_id;

  std::cout << "정보 삭제 : " << std::endl;

  std::cout << "거래 아이디";
  std::cin >> transaction_id;

  transaction_info = this->bookKeeper->Read(transaction_id);

  if (!transaction_info.has_value()) {
    std::cout << "이미 삭제되었거나 잘못된 거래 데이터입니다!" << std::endl;
    return;
  }

  std::cout << "삭제될 데이터는 다음과 같습니다 : " << std::endl;

  this->ShowTransaction(transaction_info.value());

  std::string confirm;

  std::cout << "삭제하시겠습니까? [예/아니요]";
  std::cin >> confirm;

  if (!std::equal(confirm.begin(), confirm.end(), "예")) {
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

  std::cout << "차변 : " << settlement->total_credit << std::endl;
  std::cout << "대변 : " << settlement->total_debit << std::endl;

  system("pause");
}

void BookKeeperUI::StartConsoleApp() {
  int menu_selection;

  while (true) {
    this->ShowMenu();
    std::cout << "입력 : ";

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
        std::cout << "잘못 입력하셨습니다. 다시 시도하세요!" << std::endl;
        std::cin.ignore();
    }
  }
}

}  // namespace book_keeper