#include <gtest/gtest.h>
#include <sstream>
#include "../src/distributedv2/context.h"

using namespace dgl::distributedv2;

TEST(DistV2DeserializeTest, STREAM) {
  std::stringstream ss;
  ss.write("foo", 3);
  ss.write("bar", 3);
  char buf[7] = {0};
  ASSERT_EQ(ss.rdbuf()->sgetn(buf,6), 6);
  ASSERT_STREQ(buf, "foobar");
}

TEST(DistV2DeserializeTest, STREAM_PROG) {
  std::stringstream ss;
  ss.write("foo", 3);
  ss.write("bar", 3);
  char buf[7] = {0};
  ASSERT_EQ(ss.rdbuf()->sgetn(buf,7), 6);
  ASSERT_STREQ(buf, "foobar");
  ss.seekg((int)ss.tellg() - 6);
  ASSERT_EQ(ss.rdbuf()->sgetn(buf,7), 6);
  ASSERT_STREQ(buf, "foobar");
}

TEST(DistV2DeserializeTest, STREAM_MORE) {
  std::stringstream ss;
  char buf[4] = {0};
  for (size_t trial = 0; trial < 100000LL; trial++) {
    ss.write("foo", 3);
    ASSERT_EQ(ss.rdbuf()->sgetn(buf, 3), 3);
    ASSERT_STREQ(buf, "foo");
  }
}

// latency: 1~2 [us/trial]
TEST(DistV2DeserializeTest, STREAM_RESET) {
  std::stringstream ss;
  char buf[4] = {0};
  for (size_t trial = 0; trial < 100000LL; trial++) {
    ss.write("foo", 3);
    ASSERT_EQ(ss.rdbuf()->sgetn(buf, 3), 3);
    ASSERT_STREQ(buf, "foo");
    std::string str = ss.str();
    ss.seekg(0);
    ss.seekp(0);
    ss.write(str.c_str(), str.size());
  }
}


TEST(DistV2DeserializeTest, VALID) {
  ServiceManager sm(0, 1);
  EndpointState es(0);
  const ServiceManager::stream_len_t len = 3;
  const ServiceManager::stream_sid_t sid = 0;
  const char buf[] = "Hi";
  const ServiceManager::stream_term_t term = ServiceManager::TERM;

  es.ss.write((const char *)&len, sizeof(ServiceManager::stream_len_t));
  es.ss.write((const char *)&sid, sizeof(ServiceManager::stream_sid_t));
  es.ss.write((const char *)buf, 3);
  es.ss.write((const char *)&term, sizeof(ServiceManager::stream_term_t));
  int c = sm.deserialize(&es);
  ASSERT_EQ(es.sid, 0);
  ASSERT_EQ(es.len, 3);
  ASSERT_EQ(es.sstate, StreamState::TERM);
  ASSERT_EQ(c, 1);
  char buf2[3];
  ASSERT_EQ(es.ss.rdbuf()->sgetn(buf2, 3), 3);
  ASSERT_STREQ(buf2, "Hi");
  c = sm.deserialize(&es);
  ASSERT_EQ(es.sstate, StreamState::LEN);
  ASSERT_EQ(c, 0);
}

TEST(DistV2DeserializeTest, PROG) {
  ServiceManager sm(0, 1);
  EndpointState es(0);
  const ServiceManager::stream_len_t len = 3;
  const ServiceManager::stream_sid_t sid = 0;
  const char buf[] = "Hi";
  es.ss.write((const char *)&len, sizeof(ServiceManager::stream_len_t));
  es.ss.write((const char *)&sid, sizeof(ServiceManager::stream_sid_t));
  es.ss.write((const char *)buf, 3);
  int c = sm.deserialize(&es);
  ASSERT_EQ(es.sid, 0);
  ASSERT_EQ(es.len, 3);
  ASSERT_EQ(es.sstate, StreamState::TERM);
  ASSERT_EQ(c, 1);
  char buf2[3];
  ASSERT_EQ(es.ss.rdbuf()->sgetn(buf2, 3), 3);
  c = sm.deserialize(&es);
  ASSERT_EQ(es.sstate, StreamState::TERM);
  ASSERT_EQ(c, 0);
}


TEST(DistV2DeserializeTest, VALID2) {
  ServiceManager sm(0, 1);
  EndpointState es(0);
  const ServiceManager::stream_len_t len = 3;
  const ServiceManager::stream_sid_t sid = 0;
  const char buf[] = "Hi";
  char buf2[3];
  const ServiceManager::stream_term_t term = ServiceManager::TERM;

  for (size_t trial = 0; trial < 100000LL; trial++) {
    es.ss.write((const char *)&len, sizeof(ServiceManager::stream_len_t));
    es.ss.write((const char *)&sid, sizeof(ServiceManager::stream_sid_t));
    es.ss.write((const char *)buf, 3);
    es.ss.write((const char *)&term, sizeof(ServiceManager::stream_term_t));
  }
  for (size_t trial = 0; trial < 100000LL; trial++) {
    int c = sm.deserialize(&es);
    ASSERT_EQ(c, 1);
    ASSERT_EQ(es.sid, 0);
    ASSERT_EQ(es.len, 3);
    ASSERT_EQ(es.sstate, StreamState::TERM);
    ASSERT_EQ(es.ss.rdbuf()->sgetn(buf2, 3), 3);
  }
  int c = sm.deserialize(&es);
  ASSERT_EQ(es.sstate, StreamState::LEN);
  ASSERT_EQ(c, 0);
}

TEST(DistV2DeserializeTest, SERIALIZE) {
  ServiceManager sm(0, 1);
  EndpointState es(0);
  const ServiceManager::stream_len_t len = 3;
  const ServiceManager::stream_sid_t sid = 0;
  const char buf[] = "Hi";
  std::string msg = ServiceManager::serialize(sid, buf, len);
  es.ss.write(msg.c_str(), msg.size());
  int c = sm.deserialize(&es);
  ASSERT_EQ(es.sid, 0);
  ASSERT_EQ(es.len, 3);
  ASSERT_EQ(es.sstate, StreamState::TERM);
  ASSERT_EQ(c, 1);
  char buf2[3];
  ASSERT_EQ(es.ss.rdbuf()->sgetn(buf2, 3), 3);
  ASSERT_STREQ(buf2, "Hi");
  c = sm.deserialize(&es);
  ASSERT_EQ(es.sstate, StreamState::LEN);
  ASSERT_EQ(c, 0);
}
