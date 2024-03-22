/*
 * Created by Minseok Chu on 12/29/23.
 */

#ifndef BOOKKEEPER_BOOK_KEEPER_UI_H
#define BOOKKEEPER_BOOK_KEEPER_UI_H

#include <iostream>
#include <sstream>

#ifdef WIN32
#include <optional>
#endif

#include "book_keeper.h"

namespace book_keeper {

class BookKeeperUI {
 public:
  BookKeeperUI(BookKeeper* bookKeeper);

  void StartConsoleApp();
 private:
  void ShowMenu();
  void ShowTransactions(int row_begin, int row_end);
  void ShowTransaction(book_keeper::TransactionInfo& transactionInfo);
  void ShowSettlementDetails();
  void AddTransactionFromConsole();
  void ModifyTransactionFromConsole();
  void DeleteTransactionFromConsole();
  void ClearConsole();

  BookKeeper* bookKeeper;
};

}  // namespace book_keeper

#endif  // BOOKKEEPER_BOOK_KEEPER_UI_H
