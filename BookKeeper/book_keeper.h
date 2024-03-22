/*
 * Created by Minseok Chu on 12/29/23.
 */

#ifndef BOOKKEEPER_BOOK_KEEPER_H
#define BOOKKEEPER_BOOK_KEEPER_H

#include <iostream>

#ifdef WIN32
#include <optional>
#endif

#include "linked_list.h"

namespace book_keeper {

struct TransactionInfo {
  int id;
  int left_entry;
  int right_entry;
  std::string left_entry_comment;
  std::string right_entry_comment;
  std::string transaction_comment;
};

struct Settlement {
  int total_credit;
  int total_debit;
};

class BookKeeper {
 public:
  BookKeeper();

  std::optional<TransactionInfo> Read(int id);
  std::optional<TransactionInfo> ReadByIndex(int index);
  std::optional<Settlement> GetSettlement();
  void Create(TransactionInfo const& transactionInfo);
  errno_t Update(TransactionInfo const& transactionInfo);
  errno_t Delete(int id);
  void Sort();
  int GetSize();
 private:
  linked_list::SimpleList* transaction_list_;
  int transaction_count_;
};

}  // namespace book_keeper

#endif  // BOOKKEEPER_BOOK_KEEPER_H
