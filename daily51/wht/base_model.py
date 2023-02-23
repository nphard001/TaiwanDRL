class BaseModel:

    def __call__(self, key, default=None):
        """ sugar function of self.param.get(key, default) """
        if not hasattr(self, 'param'):
            self.param = {}
        return self.param.get(key, default)

    def set_param(self, input, mode="hard"):
        """
        update param with ``input`` (input=None means no update)
        ``mode="hard"`` like default update
        ``mode="soft"`` will update exist keys only and ignore unmatched input key
        ``mode="strict"`` will raise ``KeyError`` when there is unmatched input key
        """
        if not hasattr(self, 'param'):
            self.param = {}
        if input is None:
            return self

        for k in input:
            if mode=="hard" or k in self.param:
                self.param[k] = input[k]
            elif mode=="strict":
                raise KeyError(f"{k} is redundant, set_param failed. ({self.__class__.__name__})")

        return self
